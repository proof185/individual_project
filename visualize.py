"""Visualization utilities for motion sequences."""

from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
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


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """
    Recover 3D joint positions from root-relative representation.
    
    Args:
        data: Motion data in RIC format (T, F)
        joints_num: Number of joints
    
    Returns:
        joints: 3D joint positions (T, J, 3)
    """
    data = np.asarray(data, dtype=np.float32)
    T = data.shape[0]
    
    # Root rotation velocity
    r_rot_vel = data[:, 0]
    r_rot_ang = np.zeros(T, dtype=np.float32)
    r_rot_ang[1:] = np.cumsum(r_rot_vel[:-1])
    
    # Root position
    r_pos = np.zeros((T, 3), dtype=np.float32)
    r_pos[:, 1] = data[:, 3]  # Height
    
    # Root velocity in local frame
    r_vel_local = np.zeros((T, 3), dtype=np.float32)
    r_vel_local[1:, 0] = data[:-1, 1]
    r_vel_local[1:, 2] = data[:-1, 2]
    
    # Transform to world frame
    cos_r = np.cos(r_rot_ang)
    sin_r = np.sin(r_rot_ang)
    r_vel_world = np.zeros_like(r_vel_local)
    r_vel_world[:, 0] = cos_r * r_vel_local[:, 0] - sin_r * r_vel_local[:, 2]
    r_vel_world[:, 2] = sin_r * r_vel_local[:, 0] + cos_r * r_vel_local[:, 2]
    
    # Integrate to get position
    r_pos[:, 0] = np.cumsum(r_vel_world[:, 0])
    r_pos[:, 2] = np.cumsum(r_vel_world[:, 2])
    
    # Relative joint coordinates
    ric = data[:, 4:4 + (joints_num - 1) * 3]
    ric = ric.reshape(T, joints_num - 1, 3)
    
    # Rotate to world frame
    positions = np.zeros((T, joints_num - 1, 3), dtype=np.float32)
    positions[:, :, 0] = cos_r[:, None] * ric[:, :, 0] - sin_r[:, None] * ric[:, :, 2]
    positions[:, :, 1] = ric[:, :, 1]
    positions[:, :, 2] = sin_r[:, None] * ric[:, :, 0] + cos_r[:, None] * ric[:, :, 2]
    
    # Add root position
    positions[:, :, 0] += r_pos[:, 0:1]
    positions[:, :, 2] += r_pos[:, 2:3]
    
    # Combine root and other joints
    joints = np.concatenate([r_pos[:, None, :], positions], axis=1)
    return joints


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
    
    # Compute bounds
    mins = motion.reshape(-1, 3).min(axis=0)
    maxs = motion.reshape(-1, 3).max(axis=0)
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
    pts = ax.scatter([], [], [], s=20)
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
        
        for k, (i, j) in enumerate(edges):
            lines[k].set_data([xs[i], xs[j]], [ys[i], ys[j]])
            lines[k].set_3d_properties([zs[i], zs[j]])
            lines[k].set_color(color)
        
        kf_str = " [KEYFRAME]" if is_keyframe else ""
        ax.set_title(f"{title} | frame {t+1}/{T}{kf_str}")
        return [pts] + lines
    
    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)
    plt.close(fig)
    return anim


def visualize_motion_file(
    filepath: str,
    title: str = "Motion",
    stride: int = 1,
    keyframe_indices: Optional[List[int]] = None,
    show_html: bool = True,
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
    joints = recover_from_ric(vecs, joints_num=22)
    print(f"Loaded motion: {joints.shape}")
    
    anim = animate_skeleton(
        joints,
        edges=HUMANML3D_EDGES,
        title=title,
        stride=stride,
        center=True,
        keyframe_indices=keyframe_indices,
    )
    
    if show_html:
        return HTML(anim.to_jshtml())
    return anim


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <motion_file.npy>")
        return
    
    filepath = sys.argv[1]
    anim = visualize_motion_file(filepath, show_html=False)
    plt.show()


if __name__ == '__main__':
    main()
