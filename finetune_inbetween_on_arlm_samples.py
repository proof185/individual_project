"""Prepare ARLM-generated samples and fine-tune in-between diffusion.

This script:
1) Converts TRAIN_ALL_GEN-style `*_pred.npy` files into HumanML3D
   conditioning format (`<sample_id>.npy`).
2) Launches in-between diffusion fine-tuning from a resume checkpoint.
"""

import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import types

import numpy as np
import torch


def _load_humanml_preprocess(t2mgpt_root: str):
    t2mgpt_root = os.path.abspath(t2mgpt_root)
    utils_dir = os.path.join(t2mgpt_root, "utils")
    if not os.path.isdir(utils_dir):
        raise FileNotFoundError(f"Missing T2M-GPT utils directory: {utils_dir}")

    # Compatibility for older utils code on newer NumPy versions.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    def _load_module(module_name: str, file_path: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec for {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # Force a package-like `utils` namespace mapped to T2M-GPT.
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [utils_dir]  # type: ignore[attr-defined]
    sys.modules["utils"] = utils_pkg

    quaternion_mod = _load_module("utils.quaternion", os.path.join(utils_dir, "quaternion.py"))
    param_mod = _load_module("utils.paramUtil", os.path.join(utils_dir, "paramUtil.py"))
    skeleton_mod = _load_module("utils.skeleton", os.path.join(utils_dir, "skeleton.py"))

    t2m_kinematic_chain = param_mod.t2m_kinematic_chain
    t2m_raw_offsets = param_mod.t2m_raw_offsets
    qbetween_np = quaternion_mod.qbetween_np
    qfix = quaternion_mod.qfix
    qinv_np = quaternion_mod.qinv_np
    qmul_np = quaternion_mod.qmul_np
    qrot_np = quaternion_mod.qrot_np
    quaternion_to_cont6d_np = quaternion_mod.quaternion_to_cont6d_np
    Skeleton = skeleton_mod.Skeleton

    return {
        "Skeleton": Skeleton,
        "t2m_kinematic_chain": t2m_kinematic_chain,
        "t2m_raw_offsets": t2m_raw_offsets,
        "qbetween_np": qbetween_np,
        "qfix": qfix,
        "qinv_np": qinv_np,
        "qmul_np": qmul_np,
        "qrot_np": qrot_np,
        "quaternion_to_cont6d_np": quaternion_to_cont6d_np,
    }


def _to_motion_array(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[2] == 22 and arr.shape[3] == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported motion array shape at {path}: {arr.shape}")


def _build_xyz_to_hml_converter(humanml_root: str, preprocess_lib: dict):
    Skeleton = preprocess_lib["Skeleton"]
    t2m_raw_offsets = preprocess_lib["t2m_raw_offsets"]
    t2m_kinematic_chain = preprocess_lib["t2m_kinematic_chain"]
    qbetween_np = preprocess_lib["qbetween_np"]
    qfix = preprocess_lib["qfix"]
    qinv_np = preprocess_lib["qinv_np"]
    qmul_np = preprocess_lib["qmul_np"]
    qrot_np = preprocess_lib["qrot_np"]
    quaternion_to_cont6d_np = preprocess_lib["quaternion_to_cont6d_np"]

    l_idx1, l_idx2 = 5, 8
    fid_r, fid_l = [8, 11], [7, 10]
    face_joint_indx = [2, 1, 17, 16]
    joints_num = 22
    feet_thre = 0.002

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    tgt_skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, "cpu")

    joints_dir = os.path.join(humanml_root, "joints")
    ref_path = os.path.join(joints_dir, "000021.npy")
    if not os.path.exists(ref_path):
        refs = sorted(glob.glob(os.path.join(joints_dir, "*.npy")))
        if not refs:
            raise FileNotFoundError(f"No reference joints found in {joints_dir}")
        ref_path = refs[0]

    ref = np.load(ref_path).astype(np.float32)
    if ref.ndim == 4 and ref.shape[0] == 1:
        ref = ref[0]
    if ref.ndim != 3 or ref.shape[2] != 3 or ref.shape[1] < 22:
        raise ValueError(f"Unexpected reference joints shape at {ref_path}: {ref.shape}")
    ref = ref[:, :22, :]
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(ref[0]))

    def uniform_skeleton(positions: np.ndarray) -> np.ndarray:
        src_skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, "cpu")
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
        tgt_offset = tgt_offsets.numpy()

        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
        scale_rt = tgt_leg_len / max(src_leg_len, 1e-8)

        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        src_skel.set_offset(tgt_offsets)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints

    def process_file(positions: np.ndarray) -> np.ndarray:
        positions = uniform_skeleton(positions)

        floor_height = positions.min(axis=0).min(axis=0)[1]
        positions[:, :, 1] -= floor_height

        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
        positions = positions - root_pose_init_xz

        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / (np.sqrt((across**2).sum(axis=-1))[..., np.newaxis] + 1e-8)

        forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
        forward_init = forward_init / (np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis] + 1e-8)

        target = np.array([[0, 0, 1]], dtype=np.float32)
        root_quat_init = qbetween_np(forward_init, target)
        root_quat_init = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init
        positions = qrot_np(root_quat_init, positions)

        global_positions = positions.copy()

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < feet_thre).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < feet_thre).astype(np.float32)

        skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
        quat_params = qfix(quat_params)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)

        r_rot = quat_params[:, 0].copy()
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity_quat = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

        # Rotation-invariant local joint positions.
        local_pos = positions.copy()
        local_pos[..., 0] -= local_pos[:, 0:1, 0]
        local_pos[..., 2] -= local_pos[:, 0:1, 2]
        local_pos = qrot_np(np.repeat(r_rot[:, None], local_pos.shape[1], axis=1), local_pos)

        root_y = local_pos[:, 0, 1:2]
        r_velocity = np.arcsin(np.clip(r_velocity_quat[:, 2:3], -1.0, 1.0))
        l_velocity = velocity[:, [0, 2]]
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
        ric_data = local_pos[:, 1:].reshape(len(local_pos), -1)

        local_vel = qrot_np(
            np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
            global_positions[1:] - global_positions[:-1],
        )
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = root_data
        data = np.concatenate([data, ric_data[:-1]], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        data = np.concatenate([data, local_vel], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)
        return data.astype(np.float32)

    return process_file


def _convert_pred_samples(
    sample_dir: str,
    output_dir: str,
    overwrite: bool,
    xyz_to_hml_converter,
) -> tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    pred_files = sorted(glob.glob(os.path.join(sample_dir, "*_pred.npy")))

    converted = 0
    skipped = 0
    for pred_path in pred_files:
        name = os.path.basename(pred_path)
        sample_id = name[:-9]  # strip `_pred.npy`
        out_path = os.path.join(output_dir, f"{sample_id}.npy")

        # Always refresh converted conditioning motions so fine-tuning uses
        # current TRAIN_ALL_GEN predictions rather than stale cached files.
        if os.path.exists(out_path):
            os.remove(out_path)

        motion = _to_motion_array(pred_path)
        if motion.ndim == 3 and motion.shape[1:] == (22, 3):
            motion = xyz_to_hml_converter(motion)
        elif motion.ndim != 2:
            raise ValueError(f"Unsupported converted motion shape for {pred_path}: {motion.shape}")

        if motion.shape[1] != 263:
            raise ValueError(
                f"Converted motion must be (T, 263) for HumanML3D, got {motion.shape} at {pred_path}"
            )

        np.save(out_path, motion)
        converted += 1

    return converted, skipped


def _count_train_ids(humanml_root: str) -> int:
    split_file = os.path.join(humanml_root, "train.txt")
    if not os.path.exists(split_file):
        return -1
    with open(split_file, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def _validate_prepared_feature_dir(prepared_dir: str, feature_dim: int = 263) -> int:
    feature_files = sorted(glob.glob(os.path.join(prepared_dir, "*.npy")))
    if not feature_files:
        raise FileNotFoundError(f"No prepared feature files found in {prepared_dir}")

    for path in feature_files:
        motion = np.load(path)
        if motion.ndim == 3 and motion.shape[0] == 1:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[1] != feature_dim:
            raise ValueError(
                f"Prepared conditioning file must be HumanML3D features with shape (T, {feature_dim}), got {motion.shape} at {path}"
            )

    return len(feature_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune in-between diffusion on ARLM generated samples")
    parser.add_argument("--project-root", type=str, default=".", help="Path to individual_project root")
    parser.add_argument("--humanml-root", type=str, default="humanml", help="Path to HumanML3D root")
    parser.add_argument("--sample-dir", type=str, default="TRAIN_ALL_GEN", help="Directory containing *_pred.npy ARLM samples")
    parser.add_argument("--t2mgpt-root", type=str, default="D:/Projects/T2M-GPT", help="Path to T2M-GPT repo for XYZ->feature conversion utils")
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="humanml/arlm_train_pred_joint_vecs",
        help="Directory to write converted conditioning motions as <id>.npy",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default="checkpoints/composite_inbetween_step100000.pt",
        help="In-between checkpoint to resume from",
    )
    parser.add_argument("--start-step", type=int, default=100000, help="Expected start step of resume checkpoint")
    parser.add_argument("--finetune-steps", type=int, default=20000, help="Additional fine-tuning steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Reduced fine-tuning learning rate")
    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="fine_tuned_inbetweeening",
        help="Checkpoint prefix for outputs: checkpoints/<prefix>_stepN.pt",
    )
    parser.add_argument("--overwrite-converted", action="store_true", help="Overwrite existing converted <id>.npy files")
    parser.add_argument("--disable-selector", action="store_true", help="Disable learned keyframe selector")
    parser.add_argument(
        "--prepared-only",
        action="store_true",
        help="Skip XYZ conversion and fine-tune directly from --prepared-dir, which must already contain HumanML3D feature motions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = os.path.abspath(args.project_root)
    humanml_root = os.path.abspath(os.path.join(project_root, args.humanml_root))
    t2mgpt_root = os.path.abspath(args.t2mgpt_root)
    prepared_dir = os.path.abspath(os.path.join(project_root, args.prepared_dir))
    resume_ckpt = os.path.abspath(os.path.join(project_root, args.resume_ckpt))
    train_py = os.path.join(project_root, "train.py")

    if not os.path.exists(train_py):
        raise FileNotFoundError(f"Missing train.py: {train_py}")
    if not os.path.exists(humanml_root):
        raise FileNotFoundError(f"Missing HumanML3D root: {humanml_root}")
    if not args.prepared_only and not os.path.exists(t2mgpt_root):
        raise FileNotFoundError(f"Missing T2M-GPT root: {t2mgpt_root}")
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Missing resume checkpoint: {resume_ckpt}")

    if args.prepared_only:
        available = _validate_prepared_feature_dir(prepared_dir)
        converted = 0
        skipped = 0
        print(f"Using pre-generated HumanML3D conditioning features: files={available}, dir={prepared_dir}")
    else:
        sample_dir = os.path.abspath(os.path.join(project_root, args.sample_dir))
        if not os.path.exists(sample_dir):
            raise FileNotFoundError(f"Missing ARLM sample dir: {sample_dir}")

        preprocess_lib = _load_humanml_preprocess(t2mgpt_root)
        xyz_to_hml_converter = _build_xyz_to_hml_converter(humanml_root, preprocess_lib)

        converted, skipped = _convert_pred_samples(
            sample_dir,
            prepared_dir,
            args.overwrite_converted,
            xyz_to_hml_converter,
        )
        print(f"Prepared conditioning samples: converted={converted}, skipped={skipped}, dir={prepared_dir}")

    train_count = _count_train_ids(humanml_root)
    if train_count > 0:
        available = len(glob.glob(os.path.join(prepared_dir, "*.npy")))
        print(f"HumanML3D train ids={train_count}, prepared conditioning files={available}")

    total_steps = int(args.start_step + args.finetune_steps)
    cmd = [
        sys.executable,
        train_py,
        "--stage",
        "inbetween",
        "--inbetween-resume",
        resume_ckpt,
        "--inbetween-steps",
        str(total_steps),
        "--inbetween-ckpt-prefix",
        args.ckpt_prefix,
        "--keyframe-source-dir",
        prepared_dir,
        "--lr",
        str(float(args.lr)),
    ]
    if args.disable_selector:
        cmd.append("--disable-selector")

    print("Launching fine-tune:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
