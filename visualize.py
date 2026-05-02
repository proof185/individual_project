"""Visualization utilities for motion sequences."""

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
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


def recover_from_ric(data: np.ndarray, joints_num: int = 22, add_root_y_to_joints: bool = False) -> np.ndarray:
    """
    Recover 3D joint positions from root-relative representation.
    
    Args:
        data: Motion data in RIC format (T, F)
        joints_num: Number of joints
    
    Returns:
        joints: 3D joint positions (T, J, 3)
    """
    data_t = torch.as_tensor(np.asarray(data, dtype=np.float32))

    rot_vel = data_t[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data_t.shape[:-1] + (4,), dtype=data_t.dtype, device=data_t.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data_t.shape[:-1] + (3,), dtype=data_t.dtype, device=data_t.device)
    r_pos[..., 1:, [0, 2]] = data_t[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data_t[..., 3]

    positions = data_t[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = _qrot(_qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    if add_root_y_to_joints:
        positions[..., 1] += r_pos[..., 1:2]
    joints = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
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
        motion = motion - motion[:, [0], :]
    
    # Compute plot bounds. Robust percentile bounds avoid tiny-looking skeletons
    # when a few outlier frames have implausible coordinates.
    flat = motion.reshape(-1, 3)
    # Always use 5-95 percentile for robust bounds when reconstructing from features
    mins = np.percentile(flat, 5.0, axis=0)
    maxs = np.percentile(flat, 95.0, axis=0)
    span = (maxs - mins).max()
    center_pt = (maxs + mins) / 2
    half = span / 2 * 1.2
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center_pt[0] - half, center_pt[0] + half)
    ax.set_ylim(center_pt[1] - half, center_pt[1] + half)
    ax.set_zlim(center_pt[2] - half, center_pt[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Initialize plot elements
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
        
        # Color based on keyframe
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


def visualize_motion_file(
    filepath: str,
    title: str = "Motion",
    stride: int = 1,
    keyframe_indices: Optional[List[int]] = None,
    show_html: bool = True,
    center: bool = True,
    bounds_percentile: float = 1.0,
    add_root_y_to_joints: bool = False,
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
        joints = recover_from_ric(vecs, joints_num=22, add_root_y_to_joints=add_root_y_to_joints)
        print(f"Loaded motion (recovered from features): {joints.shape}")
    
    anim = animate_skeleton(
        joints,
        edges=HUMANML3D_EDGES,
        title=title,
        stride=stride,
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
    parser.add_argument(
        "--add-root-y",
        action="store_true",
        help="Add root Y to non-root joints during RIC decode (debug for AR feature conventions)",
    )
    args = parser.parse_args()

    anim = visualize_motion_file(
        args.motion_file,
        show_html=False,
        add_root_y_to_joints=bool(args.add_root_y),
    )
    plt.show()


if __name__ == '__main__':
    main()
