import contextlib
import importlib
import json
import os
import sys
import types

import numpy as np
import torch

if not hasattr(np, 'float'):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, 'int'):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore[attr-defined]


_CONFLICTING_IMPORT_PREFIXES = (
    'utils',
    'model',
    'diffusion',
    'data_loaders',
    'configs',
    'sample',
)

_AUXILIARY_MODULE_STUBS = (
    'spacy',
)


def looks_like_condmdi_checkpoint(checkpoint_path: str) -> bool:
    args_path = os.path.join(os.path.dirname(checkpoint_path), 'args.json')
    if not os.path.exists(checkpoint_path) or not os.path.exists(args_path):
        return False
    try:
        with open(args_path, 'r', encoding='utf-8') as f:
            args = json.load(f)
    except Exception:
        return False
    return bool(args.get('dataset') == 'humanml' and args.get('keyframe_conditioned'))


def infer_condmdi_root(checkpoint_path: str) -> str:
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    if os.path.basename(save_dir).lower() != 'save':
        raise ValueError(f'Could not infer CondMDI root from checkpoint: {checkpoint_path}')
    return os.path.dirname(save_dir)


@contextlib.contextmanager
def _temporary_condmdi_imports(condmdi_root: str):
    previous_cwd = os.getcwd()
    inserted_path = False
    removed_modules: dict[str, object] = {}
    package_paths = {
        name: os.path.join(condmdi_root, name)
        for name in _CONFLICTING_IMPORT_PREFIXES
        if os.path.isdir(os.path.join(condmdi_root, name))
    }

    for module_name in list(sys.modules.keys()):
        if module_name in _CONFLICTING_IMPORT_PREFIXES or module_name.startswith(tuple(f'{name}.' for name in _CONFLICTING_IMPORT_PREFIXES)):
            removed_modules[module_name] = sys.modules.pop(module_name)
        elif module_name in _AUXILIARY_MODULE_STUBS:
            removed_modules[module_name] = sys.modules.pop(module_name)

    if condmdi_root not in sys.path:
        sys.path.insert(0, condmdi_root)
        inserted_path = True

    os.chdir(condmdi_root)
    try:
        for package_name, package_path in package_paths.items():
            package_module = types.ModuleType(package_name)
            package_module.__path__ = [package_path]  # type: ignore[attr-defined]
            sys.modules[package_name] = package_module
        for module_name in _AUXILIARY_MODULE_STUBS:
            sys.modules[module_name] = types.ModuleType(module_name)
        yield
    finally:
        os.chdir(previous_cwd)
        for module_name in list(sys.modules.keys()):
            if module_name in _CONFLICTING_IMPORT_PREFIXES or module_name.startswith(tuple(f'{name}.' for name in _CONFLICTING_IMPORT_PREFIXES)):
                sys.modules.pop(module_name, None)
            elif module_name in _AUXILIARY_MODULE_STUBS:
                sys.modules.pop(module_name, None)
        sys.modules.update(removed_modules)
        if inserted_path:
            sys.path.remove(condmdi_root)


def _build_obs_mask(valid_mask: torch.Tensor, keyframe_indices: torch.Tensor, keyframe_mask: torch.Tensor, feature_dim: int) -> torch.Tensor:
    batch_size, time_steps = valid_mask.shape
    obs_mask = torch.zeros(batch_size, feature_dim, 1, time_steps, dtype=torch.bool, device=valid_mask.device)
    for batch_idx in range(batch_size):
        valid_kf = keyframe_mask[batch_idx].bool()
        if not valid_kf.any():
            continue
        frame_ids = keyframe_indices[batch_idx][valid_kf].long()
        frame_ids = frame_ids[(frame_ids >= 0) & (frame_ids < time_steps)]
        if frame_ids.numel() == 0:
            continue
        allowed = valid_mask[batch_idx, frame_ids]
        frame_ids = frame_ids[allowed]
        if frame_ids.numel() == 0:
            continue
        obs_mask[batch_idx, :, 0, frame_ids] = True
    return obs_mask


class ExternalCondMDIDiffusionAdapter:
    def __init__(
        self,
        model,
        diffusion,
        condmdi_root: str,
        local_mean: torch.Tensor,
        local_std: torch.Tensor,
        abs_mean: torch.Tensor,
        abs_std: torch.Tensor,
        device: torch.device,
        max_frames: int,
        oracle_model=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.oracle_model = oracle_model if oracle_model is not None else model
        self.condmdi_root = condmdi_root
        self.local_mean = local_mean.to(device).view(1, -1, 1, 1)
        self.local_std = local_std.to(device).view(1, -1, 1, 1)
        self.abs_mean = abs_mean.to(device).view(1, -1, 1, 1)
        self.abs_std = abs_std.to(device).view(1, -1, 1, 1)
        self.device = device
        self.max_frames = max_frames

    @staticmethod
    def _qinv(q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[..., 1:] = -out[..., 1:]
        return out

    @staticmethod
    def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qvec = q[..., 1:]
        uv = torch.cross(qvec, v, dim=-1)
        uuv = torch.cross(qvec, uv, dim=-1)
        return v + 2 * (q[..., :1] * uv + uuv)

    def _relative_to_absolute_root(self, data: torch.Tensor) -> torch.Tensor:
        # data: [B, F, 1, T] in denormalized relative HumanML3D feature space
        output = data.clone()
        seq = output.permute(0, 2, 3, 1)
        rot_vel = seq[..., 0]

        rot_ang = torch.zeros_like(rot_vel)
        rot_ang[..., 1:] = rot_vel[..., :-1]
        rot_ang = torch.cumsum(rot_ang, dim=-1)

        r_rot_quat = torch.zeros(seq.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
        r_rot_quat[..., 0] = torch.cos(rot_ang)
        r_rot_quat[..., 2] = torch.sin(rot_ang)

        r_pos = torch.zeros(seq.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
        r_pos[..., 1:, [0, 2]] = seq[..., :-1, 1:3]
        r_pos = self._qrot(self._qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., 1] = seq[..., 3]

        output[:, :1] = rot_ang.unsqueeze(1).unsqueeze(2)
        output[:, 1:2] = r_pos[..., 0].unsqueeze(1).unsqueeze(2)
        output[:, 2:3] = r_pos[..., 2].unsqueeze(1).unsqueeze(2)
        return output

    def _absolute_to_relative_root(self, data: torch.Tensor) -> torch.Tensor:
        # data: [B, F, 1, T] in denormalized abs-3d HumanML3D feature space
        output = data.clone()
        seq = output.permute(0, 2, 3, 1)
        gl_pos = seq[..., 1:4][..., [0, 2, 1]]
        gl_rot = seq[..., :1]

        rel_pos = torch.zeros_like(gl_pos)
        rel_pos[:, :, 1:, [0, 2]] = gl_pos[:, :, 1:, [0, 2]] - gl_pos[:, :, :-1, [0, 2]]

        gl_quat_rot = torch.zeros(gl_rot.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
        gl_quat_rot[..., :1] = torch.cos(gl_rot)
        gl_quat_rot[..., 2:3] = torch.sin(gl_rot)
        rel_pos = self._qrot(gl_quat_rot, rel_pos)
        rel_pos[:, :, :-1] = rel_pos[:, :, 1:].clone()
        rel_pos[..., 1] = seq[..., 3]

        rel_rot = torch.zeros_like(gl_rot)
        rel_rot[:, :, :-1, :] = gl_rot[:, :, 1:, :] - gl_rot[:, :, :-1, :]

        output[:, :1] = rel_rot.permute(0, 3, 1, 2)
        output[:, 1:2] = rel_pos[..., 0].unsqueeze(1).unsqueeze(2)
        output[:, 2:3] = rel_pos[..., 2].unsqueeze(1).unsqueeze(2)
        return output

    def _normalize_local_to_abs(self, x0_local: torch.Tensor) -> torch.Tensor:
        source_rel = x0_local.permute(0, 2, 1).unsqueeze(2)
        source_rel = source_rel * (self.local_std + 1e-8) + self.local_mean
        source_abs = self._relative_to_absolute_root(source_rel)
        return (source_abs - self.abs_mean) / (self.abs_std + 1e-8)

    def _denormalize_abs_to_local(self, x_abs_norm: torch.Tensor) -> torch.Tensor:
        sampled_abs = x_abs_norm * (self.abs_std + 1e-8) + self.abs_mean
        sampled_rel = self._absolute_to_relative_root(sampled_abs)
        sampled_rel_norm = (sampled_rel - self.local_mean) / (self.local_std + 1e-8)
        return sampled_rel_norm.squeeze(2).permute(0, 2, 1).contiguous()

    def predict_x0_local(
        self,
        x0_local: torch.Tensor,
        valid_mask: torch.Tensor,
        t_batch: torch.Tensor,
        observation_mask: torch.Tensor,
        text_prompts,
    ) -> torch.Tensor:
        batch_size, time_steps, feature_dim = x0_local.shape
        if time_steps > self.max_frames:
            raise ValueError(f'Requested {time_steps} frames, but CondMDI supports at most {self.max_frames} frames in this adapter.')

        x0_local = x0_local.to(self.device).float()
        valid_mask = valid_mask.to(self.device).bool()
        t_batch = t_batch.to(self.device).long()
        observation_mask = observation_mask.to(self.device).bool()

        with _temporary_condmdi_imports(self.condmdi_root):
            x0_abs_norm = self._normalize_local_to_abs(x0_local)
            noise = torch.randn_like(x0_abs_norm)
            xt_abs = self.diffusion.q_sample(x0_abs_norm, t_batch, noise)

            padded_x0 = torch.zeros(batch_size, feature_dim, 1, self.max_frames, device=self.device)
            padded_xt = torch.zeros_like(padded_x0)
            padded_x0[:, :, :, :time_steps] = x0_abs_norm
            padded_xt[:, :, :, :time_steps] = xt_abs

            padded_valid_mask = torch.zeros(batch_size, self.max_frames, dtype=torch.bool, device=self.device)
            padded_valid_mask[:, :time_steps] = valid_mask

            padded_obs_mask = torch.zeros(batch_size, self.max_frames, dtype=torch.bool, device=self.device)
            padded_obs_mask[:, :time_steps] = observation_mask
            obs_mask = padded_obs_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, feature_dim, 1, self.max_frames).clone()

            padded_xt = torch.where(obs_mask, padded_x0, padded_xt)

            if isinstance(text_prompts, str):
                texts = [text_prompts] * batch_size
            else:
                texts = list(text_prompts)
                if len(texts) != batch_size:
                    raise ValueError(f'text_prompts list must have batch size {batch_size}, got {len(texts)}')

            model_kwargs = {
                'y': {
                    'mask': padded_valid_mask.unsqueeze(1).unsqueeze(1),
                    'lengths': valid_mask.sum(dim=1).long(),
                    'text': texts,
                },
            }
            x0_hat_abs = self.oracle_model(
                padded_xt,
                t_batch,
                model_kwargs['y'],
                obs_x0=padded_x0,
                obs_mask=obs_mask,
            )
            x0_hat_abs = x0_hat_abs[:, :, :, :time_steps]
            return self._denormalize_abs_to_local(x0_hat_abs)

    def sample_inbetween(
        self,
        model,
        shape,
        cond,
        mask,
        keyframes=None,
        keyframe_indices=None,
        keyframe_mask=None,
        guidance_scale: float = 2.5,
        cond_uncond=None,
        observation_mask=None,
        keyframe_canvas=None,
        source_motion=None,
        text_prompt=None,
    ) -> torch.Tensor:
        del model, cond, keyframes, cond_uncond, observation_mask, keyframe_canvas

        if source_motion is None:
            raise ValueError('External CondMDI inference requires source_motion so observed keyframes can be converted to abs_3d.')
        if text_prompt is None:
            raise ValueError('External CondMDI inference requires text_prompt.')
        if keyframe_indices is None or keyframe_mask is None:
            raise ValueError('External CondMDI inference requires sparse keyframe indices and mask.')

        batch_size, time_steps, feature_dim = shape
        if time_steps > self.max_frames:
            raise ValueError(f'Requested {time_steps} frames, but CondMDI supports at most {self.max_frames} frames in this adapter.')

        source_motion = source_motion.to(self.device).float()
        valid_mask = mask.to(self.device).bool()
        keyframe_indices = keyframe_indices.to(self.device)
        keyframe_mask = keyframe_mask.to(self.device)

        if source_motion.shape != (batch_size, time_steps, feature_dim):
            raise ValueError(
                f'source_motion must have shape {(batch_size, time_steps, feature_dim)}, got {tuple(source_motion.shape)}.'
            )

        with _temporary_condmdi_imports(self.condmdi_root):
            source_rel = source_motion.permute(0, 2, 1).unsqueeze(2)
            source_rel = source_rel * (self.local_std + 1e-8) + self.local_mean
            source_abs = self._relative_to_absolute_root(source_rel)
            source_abs_norm = (source_abs - self.abs_mean) / (self.abs_std + 1e-8)

            padded_source = torch.zeros(batch_size, feature_dim, 1, self.max_frames, device=self.device)
            padded_source[:, :, :, :time_steps] = source_abs_norm

            padded_valid_mask = torch.zeros(batch_size, self.max_frames, dtype=torch.bool, device=self.device)
            padded_valid_mask[:, :time_steps] = valid_mask
            lengths = valid_mask.sum(dim=1).long()
            obs_mask = _build_obs_mask(padded_valid_mask, keyframe_indices, keyframe_mask, feature_dim)

            if isinstance(text_prompt, str):
                texts = [text_prompt] * batch_size
            else:
                texts = list(text_prompt)
                if len(texts) != batch_size:
                    raise ValueError(f'text_prompt list must have batch size {batch_size}, got {len(texts)}')

            model_kwargs = {
                'y': {
                    'mask': padded_valid_mask.unsqueeze(1).unsqueeze(1),
                    'lengths': lengths,
                    'text': texts,
                    'text_scale': torch.ones(batch_size, device=self.device) * float(guidance_scale),
                    'imputate': 1,
                    'stop_imputation_at': 0,
                    'replacement_distribution': 'conditional',
                    'inpainted_motion': padded_source,
                    'inpainting_mask': obs_mask,
                    'reconstruction_guidance': False,
                },
                'obs_x0': padded_source,
                'obs_mask': obs_mask,
            }

            sampled_abs = self.diffusion.p_sample_loop(
                self.model,
                (batch_size, self.model.njoints, self.model.nfeats, self.max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sampled_abs = sampled_abs[:, :, :, :time_steps]
            sampled_abs = sampled_abs * (self.abs_std + 1e-8) + self.abs_mean
            sampled_rel = self._absolute_to_relative_root(sampled_abs)
            sampled_rel_norm = (sampled_rel - self.local_mean) / (self.local_std + 1e-8)
            return sampled_rel_norm.squeeze(2).permute(0, 2, 1).contiguous()


def load_external_condmdi_runtime(
    checkpoint_path: str,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    device: str = 'cuda',
):
    condmdi_root = infer_condmdi_root(checkpoint_path)
    target_device = torch.device(device)
    args_path = os.path.join(os.path.dirname(checkpoint_path), 'args.json')
    dataset_root = os.path.join(condmdi_root, 'dataset', 'HumanML3D')
    abs_mean = torch.from_numpy(np.load(os.path.join(dataset_root, 'Mean_abs_3d.npy'))).float().view(-1)
    abs_std = torch.from_numpy(np.load(os.path.join(dataset_root, 'Std_abs_3d.npy'))).float().view(-1)

    with open(args_path, 'r', encoding='utf-8') as f:
        model_args = json.load(f)

    with _temporary_condmdi_imports(condmdi_root):
        rotation2xyz_module = importlib.import_module('model.rotation2xyz')

        class _NoOpSMPLModel:
            def _apply(self, fn):
                return self

            def train(self, mode=True):
                return self

        class _NoOpRotation2xyz:
            def __init__(self, device, dataset='amass'):
                self.device = device
                self.dataset = dataset
                self.smpl_model = _NoOpSMPLModel()

            def __call__(self, *args, **kwargs):
                raise RuntimeError('Rotation2xyz is unavailable in the CondMDI adapter because SMPL assets are not loaded.')

        rotation2xyz_module.Rotation2xyz = _NoOpRotation2xyz

        model_util_module = importlib.import_module('utils.model_util')
        cfg_sampler_module = importlib.import_module('model.cfg_sampler')
        create_model_and_diffusion = model_util_module.create_model_and_diffusion
        load_saved_model = model_util_module.load_saved_model
        ClassifierFreeSampleModel = cfg_sampler_module.ClassifierFreeSampleModel

        args = types.SimpleNamespace(**model_args)
        dummy_data = types.SimpleNamespace(dataset=types.SimpleNamespace(num_actions=1))
        condmdi_model, condmdi_diffusion = create_model_and_diffusion(args, dummy_data)
        load_saved_model(condmdi_model, checkpoint_path)
        condmdi_model.to(target_device)
        condmdi_model.eval()

        sample_model = condmdi_model
        if getattr(args, 'cond_mask_prob', 0) > 0:
            sample_model = ClassifierFreeSampleModel(condmdi_model).to(target_device)
            sample_model.eval()
        if target_device.type == 'cpu':
            condmdi_model.float()
            sample_model.float()
            if hasattr(condmdi_model, 'clip_model'):
                condmdi_model.clip_model.float()
            if hasattr(sample_model, 'clip_model'):
                sample_model.clip_model.float()

    max_frames = 196 if getattr(args, 'dataset', 'humanml') in ('humanml', 'kit') else int(getattr(args, 'num_frames', 196))
    adapter = ExternalCondMDIDiffusionAdapter(
        model=sample_model,
        diffusion=condmdi_diffusion,
        condmdi_root=condmdi_root,
        local_mean=local_mean,
        local_std=local_std,
        abs_mean=abs_mean,
        abs_std=abs_std,
        device=target_device,
        max_frames=max_frames,
        oracle_model=condmdi_model,
    )
    condmdi_model.keyframe_selector = None
    condmdi_model.is_external_condmdi = True
    return condmdi_model, adapter


def load_external_condmdi_oracle(
    checkpoint_path: str,
    local_mean: torch.Tensor,
    local_std: torch.Tensor,
    device: str = 'cuda',
):
    model, adapter = load_external_condmdi_runtime(
        checkpoint_path=checkpoint_path,
        local_mean=local_mean,
        local_std=local_std,
        device=device,
    )
    return model, adapter