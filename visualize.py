"""Visualization utilities for motion sequences."""

from typing import Optional, List

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML


# HUMANML3D skeleton structure
HUMANML3D_EDGES = [
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
]


def _qinv(q: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    original_shape = v.shape
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def _recover_root_rot_pos(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=data.dtype, device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=data.dtype, device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """
    Recover 3D joint positions from HumanML3D RIC representation.

    Args:
        data: Motion data in RIC format (T, F=263)
        joints_num: Number of joints (default 22)

    Returns:
        joints: 3D joint positions (T, J, 3)
    """
    motion = torch.as_tensor(data, dtype=torch.float32)
    squeeze_batch = motion.ndim == 2
    if squeeze_batch:
        motion = motion.unsqueeze(0)

    r_rot_quat, r_pos = _recover_root_rot_pos(motion)
    joints = motion[..., 4:(joints_num - 1) * 3 + 4]
    joints = joints.view(joints.shape[:-1] + (joints_num - 1, 3))

    joints = _qrot(
        _qinv(r_rot_quat[..., None, :]).expand(joints.shape[:-1] + (4,)),
        joints,
    )
    joints[..., 0] += r_pos[..., 0:1]
    joints[..., 2] += r_pos[..., 2:3]
    joints = torch.cat([r_pos.unsqueeze(-2), joints], dim=-2)

    if squeeze_batch:
        joints = joints.squeeze(0)
    return joints.cpu().numpy()


def animate_skeleton(
    motion: np.ndarray,
    edges: List = HUMANML3D_EDGES,
    title: str = "Motion",
    stride: int = 1,
    elev: int = 15,
    azim: int = -70,
    interval: int = 80,
    center: bool = True,
    keyframe_indices: Optional[List[int]] = None,
    bounds_percentile: float = 1.0,
):
    """
    Create an animation of a skeleton motion.

    Args:
        motion: Joint positions (T, J, 3)
        edges: List of bone connections
        title: Plot title
        stride: Frame subsampling stride
        elev: Camera elevation angle
        azim: Camera azimuth angle
        interval: Milliseconds between frames
        center: Whether to center the motion
        keyframe_indices: List of keyframe indices to highlight

    Returns:
        Animation object
    """
    motion = motion[::stride].copy()
    T, J, _ = motion.shape

    if keyframe_indices is not None:
        keyframe_indices = [i // stride for i in keyframe_indices if i // stride < T]

    if center:
        motion[:, :, 0] -= motion[:, [0], 0]
        motion[:, :, 2] -= motion[:, [0], 2]
        motion[:, :, 1] -= motion[:, :, 1].min()

    flat = motion.reshape(-1, 3)
    lo = max(0.0, float(bounds_percentile))
    hi = min(100.0, 100.0 - lo)
    mins = np.percentile(flat, lo, axis=0)
    maxs = np.percentile(flat, hi, axis=0)
    span = (maxs - mins).max()
    center_pt = (maxs + mins) / 2
    half = span / 2 * 1.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center_pt[0] - half, center_pt[0] + half)
    ax.set_ylim(center_pt[1] - half, center_pt[1] + half)
    ax.set_zlim(center_pt[2] - half, center_pt[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pts = ax.scatter([], [], [], s=22, c='royalblue')
    lines = [ax.plot([], [], [], lw=2)[0] for _ in edges]

    def init():
        pts._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return [pts] + lines

    def update(t):
        frame = motion[t]
        xs, ys, zs = frame[:, 0], frame[:, 2], frame[:, 1]
        pts._offsets3d = (xs, ys, zs)

        is_keyframe = keyframe_indices is not None and t in keyframe_indices
        color = 'red' if is_keyframe else 'blue'
        pts.set_color('red' if is_keyframe else 'royalblue')
        pts.set_sizes(np.full(J, 30 if is_keyframe else 20))

        for k, (i, j) in enumerate(edges):
            lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
            lines[k].set_3d_properties([zs[i], zs[j]])
            lines[k].set_color(color)

        kf_str = " [KEYFRAME]" if is_keyframe else ""
        ax.set_title(f"{title} | frame {t+1}/{T}{kf_str}")
        return [pts] + lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)
    return anim


def step_through_skeleton(
    motion: np.ndarray,
    edges: List = HUMANML3D_EDGES,
    title: str = "Motion",
    stride: int = 1,
    elev: int = 15,
    azim: int = -70,
    center: bool = True,
    keyframe_indices: Optional[List[int]] = None,
    bounds_percentile: float = 1.0,
):
    """Create a manual frame-by-frame viewer controlled from the keyboard.

    Controls:
        Right / D: next frame
        Left / A: previous frame
        Home: first frame
        End: last frame
    """
    motion = motion[::stride].copy()
    T, J, _ = motion.shape

    if keyframe_indices is not None:
        keyframe_indices = [i // stride for i in keyframe_indices if i // stride < T]

    if center:
        motion[:, :, 0] -= motion[:, [0], 0]
        motion[:, :, 2] -= motion[:, [0], 2]
        motion[:, :, 1] -= motion[:, :, 1].min()

    flat = motion.reshape(-1, 3)
    lo = max(0.0, float(bounds_percentile))
    hi = min(100.0, 100.0 - lo)
    mins = np.percentile(flat, lo, axis=0)
    maxs = np.percentile(flat, hi, axis=0)
    span = (maxs - mins).max()
    center_pt = (maxs + mins) / 2
    half = span / 2 * 1.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center_pt[0] - half, center_pt[0] + half)
    ax.set_ylim(center_pt[1] - half, center_pt[1] + half)
    ax.set_zlim(center_pt[2] - half, center_pt[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pts = ax.scatter([], [], [], s=22, c='royalblue')
    lines = [ax.plot([], [], [], lw=2)[0] for _ in edges]
    state = {"frame": 0}

    def draw_frame(frame_idx: int) -> None:
        frame = motion[frame_idx]
        xs, ys, zs = frame[:, 0], frame[:, 2], frame[:, 1]
        pts._offsets3d = (xs, ys, zs)

        is_keyframe = keyframe_indices is not None and frame_idx in keyframe_indices
        color = 'red' if is_keyframe else 'blue'
        pts.set_color('red' if is_keyframe else 'royalblue')
        pts.set_sizes(np.full(J, 30 if is_keyframe else 20))

        for k, (i, j) in enumerate(edges):
            lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
            lines[k].set_3d_properties([zs[i], zs[j]])
            lines[k].set_color(color)

        kf_str = " [KEYFRAME]" if is_keyframe else ""
        ax.set_title(f"{title} | frame {frame_idx+1}/{T}{kf_str}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in {'right', 'd'}:
            state['frame'] = min(T - 1, state['frame'] + 1)
        elif event.key in {'left', 'a'}:
            state['frame'] = max(0, state['frame'] - 1)
        elif event.key == 'home':
            state['frame'] = 0
        elif event.key == 'end':
            state['frame'] = T - 1
        else:
            return
        draw_frame(state['frame'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw_frame(0)
    print("Step-through controls: Right/D = next, Left/A = previous, Home = first, End = last")
    return fig


def visualize_motion_file(
    filepath: str,
    title: str = "Motion",
    stride: int = 1,
    keyframe_indices: Optional[List[int]] = None,
    show_html: bool = True,
    center: bool = True,
    bounds_percentile: float = 1.0,
    normalized: bool = False,
    mean_path: Optional[str] = None,
    std_path: Optional[str] = None,
    interval: int = 80,
    step_through: bool = False,
):
    """
    Load and visualize motion from file.
    
    Args:
        filepath: Path to .npy file with motion data
        title: Plot title
        stride: Frame subsampling stride
        keyframe_indices: List of keyframe indices to highlight
        show_html: Whether to return HTML display
    
    Returns:
        Animation object or HTML
    """
    vecs = np.load(filepath)

    # Auto-detect format
    if vecs.ndim == 4:
        # (B, T, J, 3) — T2M-GPT eval output; take first batch item
        vecs = vecs[0]

    if vecs.ndim == 3 and vecs.shape[-1] == 3:
        # Already joint positions (T, J, 3)
        joints = vecs.astype(np.float32)
        print(f"Loaded joint-positions motion: {joints.shape}")
    else:
        # HumanML3D 263-dim feature vectors (T, F)
        if normalized:
            if not mean_path or not std_path:
                raise ValueError("normalized=True requires both mean_path and std_path")
            mean = np.load(mean_path).astype(np.float32)
            std = np.load(std_path).astype(np.float32)
            vecs = vecs.astype(np.float32) * (std + 1e-8) + mean
        joints = recover_from_ric(vecs, joints_num=22)
        print(f"Loaded motion (recovered from features): {joints.shape}")
    
    if step_through:
        return step_through_skeleton(
            joints,
            edges=HUMANML3D_EDGES,
            title=title,
            stride=stride,
            center=center,
            keyframe_indices=keyframe_indices,
            bounds_percentile=bounds_percentile,
        )

    anim = animate_skeleton(
        joints,
        edges=HUMANML3D_EDGES,
        title=title,
        stride=stride,
        interval=interval,
        center=center,
        keyframe_indices=keyframe_indices,
        bounds_percentile=bounds_percentile,
    )

    if show_html:
        return HTML(anim.to_jshtml())
    return anim


def main():
    """Command-line visualization utility."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize motion .npy files")
    parser.add_argument("motion_file", type=str, help="Path to motion .npy file")
    parser.add_argument("--normalized", action="store_true", help="Interpret 2D feature tensors as normalized features and denormalize before recovery")
    parser.add_argument("--mean-path", type=str, default=None, help="Path to mean.npy used for denormalization")
    parser.add_argument("--std-path", type=str, default=None, help="Path to std.npy used for denormalization")
    parser.add_argument("--no-center", action="store_true", help="Keep global trajectory instead of centering X/Z")
    parser.add_argument("--bounds-percentile", type=float, default=1.0, help="Lower percentile used for plot bounds; upper bound uses 100-p")
    parser.add_argument("--interval-ms", type=int, default=80, help="Animation interval in milliseconds")
    parser.add_argument("--step-through", action="store_true", help="Open an interactive frame-by-frame viewer controlled with arrow keys")
    parser.add_argument("--save-mp4", type=str, default=None, help="Optional output path for saving the animation as an MP4")
    args = parser.parse_args()

    view = visualize_motion_file(
        args.motion_file,
        show_html=False,
        normalized=bool(args.normalized),
        mean_path=args.mean_path,
        std_path=args.std_path,
        center=not args.no_center,
        bounds_percentile=float(args.bounds_percentile),
        interval=int(args.interval_ms),
        step_through=bool(args.step_through),
    )

    if args.save_mp4:
        if args.step_through:
            raise ValueError("--save-mp4 cannot be combined with --step-through")
        if not animation.writers.is_available('ffmpeg'):
            raise RuntimeError(
                "MP4 export requires ffmpeg, but matplotlib cannot find it in the current environment."
            )
        fps = max(1, int(round(1000.0 / float(args.interval_ms))))
        writer = FFMpegWriter(fps=fps)
        view.save(args.save_mp4, writer=writer)
        print(f"Saved MP4: {args.save_mp4}")

    plt.show()


if __name__ == '__main__':
    main()
