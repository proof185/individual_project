"""Static ghost-trail visualisation for a motion sequence.

Renders N skeleton poses overlaid in a single image, with keyframe poses
highlighted in a contrasting colour.  Designed to produce publication-quality
figures for dissertations and papers.

Usage
-----
    python visualize_ghost.py samples/compare_selectors/jump_kick_energy_motion.npy \\
        --keyframe-indices samples/compare_selectors/jump_kick_energy_keyframe_indices.npy \\
        --out jump_kick_energy_ghost.png

Pipeline equivalent
-------------------
    python pipeline.py ghost <motion_file> [options]
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from visualize import HUMANML3D_EDGES, recover_from_ric


# ── colours ──────────────────────────────────────────────────────────────────

# Warm tan for regular ghost poses (matches the reference mannequin look)
_DEFAULT_REGULAR_COLOR = "#C8956C"
# Steel blue for keyframe poses
_DEFAULT_KEYFRAME_COLOR = "#3A6FC4"
# First / last pose slightly different shade to bookend the sequence
_ENDPOINT_COLOR = "#A0724A"

# Floor tile colours
_FLOOR_LIGHT = (0.92, 0.92, 0.92)
_FLOOR_DARK = (0.80, 0.80, 0.80)


# ── skeleton drawing ──────────────────────────────────────────────────────────

def _draw_pose(
    ax,
    frame: np.ndarray,          # (J, 3)  raw joint coords
    edges: list,
    color: str,
    alpha: float,
    bone_lw: float,
    joint_size: float,
    zorder: int = 2,
) -> None:
    """Draw one skeleton pose onto a 3D axis.

    Axis mapping matches visualize.py: plot_x=x, plot_y=z, plot_z=y (height).
    """
    xs = frame[:, 0]
    ys = frame[:, 2]   # side-to-side becomes plot Y
    zs = frame[:, 1]   # height becomes plot Z

    for i, j in edges:
        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            [zs[i], zs[j]],
            color=color,
            lw=bone_lw,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
        )

    ax.scatter(
        xs, ys, zs,
        s=joint_size,
        c=color,
        alpha=alpha,
        zorder=zorder + 1,
        depthshade=False,
        edgecolors="none",
    )


# ── checkerboard floor ────────────────────────────────────────────────────────

def _draw_floor(
    ax,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_floor: float,
    n_tiles: int = 14,
) -> None:
    """Draw a checkerboard floor at z=z_floor using Poly3DCollection."""
    xs = np.linspace(x_range[0], x_range[1], n_tiles + 1)
    ys = np.linspace(y_range[0], y_range[1], n_tiles + 1)

    verts, facecolors = [], []
    for i in range(n_tiles):
        for j in range(n_tiles):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            verts.append([
                (x0, y0, z_floor),
                (x1, y0, z_floor),
                (x1, y1, z_floor),
                (x0, y1, z_floor),
            ])
            facecolors.append(_FLOOR_LIGHT if (i + j) % 2 == 0 else _FLOOR_DARK)

    poly = Poly3DCollection(verts, facecolors=facecolors, edgecolors="none", zorder=1)
    ax.add_collection3d(poly)


# ── main render function ──────────────────────────────────────────────────────

def render_ghost_trail(
    joints: np.ndarray,                        # (T, J, 3)
    keyframe_indices: Optional[list[int]] = None,
    n_ghosts: int = 15,
    stride: Optional[int] = None,
    spread: Optional[float] = None,
    elev: float = 12.0,
    azim: float = -45.0,
    figsize: tuple[float, float] = (14.0, 5.0),
    dpi: int = 200,
    bone_lw: float = 2.2,
    joint_size: float = 16.0,
    regular_color: str = _DEFAULT_REGULAR_COLOR,
    keyframe_color: str = _DEFAULT_KEYFRAME_COLOR,
    alpha_regular: float = 0.80,
    alpha_keyframe: float = 0.95,
    show_floor: bool = True,
    floor_tiles: int = 14,
    floor_padding: float = 0.25,
    show_axes: bool = False,
    title: Optional[str] = None,
    title_fontsize: int = 11,
) -> plt.Figure:
    """Return a Figure with the ghost trail.

    Parameters
    ----------
    joints:
        (T, J, 3) array of 3-D joint positions.
    keyframe_indices:
        Frame indices to draw in *keyframe_color*.  Pass an empty list or None
        to draw everything in *regular_color*.
    n_ghosts:
        How many ghost poses to sample uniformly across the sequence.  Ignored
        when *stride* is given.
    stride:
        Sample every *stride*-th frame instead of selecting *n_ghosts* poses.
    spread:
        If given, override the natural X spacing and place ghosts at fixed
        intervals of *spread* metres apart.  Useful for in-place motions.
    elev, azim:
        Camera elevation and azimuth in degrees.
    figsize:
        Figure size (width, height) in inches.
    dpi:
        Output resolution.
    bone_lw:
        Line width for bones.
    joint_size:
        Scatter point size for joints.
    show_floor:
        Whether to draw the checkerboard floor.
    floor_tiles:
        Number of checkerboard tiles per axis.
    show_axes:
        Set True to show tick marks and axis labels.
    title:
        Optional text title rendered at the bottom of the figure.
    """
    T, J, _ = joints.shape
    kf_set = set(keyframe_indices or [])

    # Normalise height using the global minimum across ALL frames so that the
    # floor is always at y=0, regardless of which frames are selected as ghosts.
    joints = joints.copy()
    joints[:, :, 1] -= joints[:, :, 1].min()

    # ── select ghost frames ───────────────────────────────────────────────────
    if stride is not None:
        ghost_frames = list(range(0, T, stride))
    else:
        n = min(n_ghosts, T)
        ghost_frames = [int(round(i * (T - 1) / max(n - 1, 1))) for i in range(n)]
    ghost_frames = sorted(set(ghost_frames))

    # ── optional spread override ──────────────────────────────────────────────
    poses = [joints[t].copy() for t in ghost_frames]

    if spread is not None:
        for k, frame in enumerate(poses):
            offset = k * spread - (len(poses) - 1) * spread / 2.0
            frame[:, 0] = offset

    # ── compute axis bounds ───────────────────────────────────────────────────
    all_joints = np.concatenate(poses, axis=0)
    x_all = all_joints[:, 0]
    y_all = all_joints[:, 2]   # side-to-side → plot Y
    z_all = all_joints[:, 1]   # height       → plot Z

    x_pad = (x_all.max() - x_all.min()) * floor_padding + 0.3
    y_pad = (y_all.max() - y_all.min()) * floor_padding + 0.3

    x_min, x_max = x_all.min() - x_pad, x_all.max() + x_pad
    y_min, y_max = y_all.min() - y_pad, y_all.max() + y_pad
    z_floor = -0.02           # just below the feet
    z_min = z_floor
    z_max = z_all.max() + 0.15

    # ── figure / axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Disable matplotlib's automatic depth-based z-ordering so that explicit
    # zorder values are respected — this keeps the floor beneath the skeletons.
    ax.computed_zorder = False

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    if not show_axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.tick_params(labelsize=7)

    # ── floor (zorder=1, always behind skeletons) ─────────────────────────────
    if show_floor:
        _draw_floor(
            ax,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            z_floor=z_floor,
            n_tiles=floor_tiles,
        )

    # ── ghost poses ───────────────────────────────────────────────────────────
    for k, (t, frame) in enumerate(zip(ghost_frames, poses)):
        is_kf = t in kf_set
        is_endpoint = k == 0 or k == len(ghost_frames) - 1

        if is_kf:
            color = keyframe_color
            alpha = alpha_keyframe
            lw = bone_lw * 1.25
            js = joint_size * 1.4
        elif is_endpoint:
            color = _ENDPOINT_COLOR
            alpha = alpha_regular * 0.9
            lw = bone_lw
            js = joint_size
        else:
            color = regular_color
            alpha = alpha_regular
            lw = bone_lw
            js = joint_size

        _draw_pose(ax, frame, HUMANML3D_EDGES, color, alpha, lw, js, zorder=3 + k)

    # ── title ─────────────────────────────────────────────────────────────────
    if title:
        fig.text(
            0.5, 0.02, title,
            ha="center", va="bottom",
            fontsize=title_fontsize,
            color="#333333",
        )

    plt.tight_layout(pad=0.1)
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a static ghost-trail image of a motion sequence"
    )
    parser.add_argument("motion_file", type=str, help="Path to motion .npy file (HumanML3D features or (T,J,3) joint positions)")
    parser.add_argument("--keyframe-indices", type=str, default=None,
                        help="Path to keyframe_indices.npy OR comma-separated frame numbers")
    parser.add_argument("--n-ghosts", type=int, default=15,
                        help="Number of ghost poses to sample uniformly (default: 10)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Show every N-th frame instead of --n-ghosts uniform samples")
    parser.add_argument("--spread", type=float, default=None,
                        help="Force ghosts to be spaced this many metres apart in X (for in-place motions)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (.png / .pdf / .svg).  Defaults to <motion_file>_ghost.png")
    parser.add_argument("--elev", type=float, default=12.0)
    parser.add_argument("--azim", type=float, default=-45.0)
    parser.add_argument("--figsize", type=str, default="14x5",
                        help="Figure size WxH in inches, e.g. 14x5")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bone-lw", type=float, default=2.2)
    parser.add_argument("--joint-size", type=float, default=16.0)
    parser.add_argument("--regular-color", type=str, default=_DEFAULT_REGULAR_COLOR)
    parser.add_argument("--keyframe-color", type=str, default=_DEFAULT_KEYFRAME_COLOR)
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--floor-tiles", type=int, default=14)
    parser.add_argument("--show-axes", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--normalized", action="store_true",
                        help="Motion file is in normalized feature space; provide --mean-path and --std-path")
    parser.add_argument("--mean-path", type=str, default=None)
    parser.add_argument("--std-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── load motion ───────────────────────────────────────────────────────────
    data = np.load(args.motion_file)

    if data.ndim == 4:
        data = data[0]   # (B, T, J, 3) → first sample

    if data.ndim == 3 and data.shape[-1] == 3:
        joints = data.astype(np.float32)
    else:
        if args.normalized:
            if not args.mean_path or not args.std_path:
                raise ValueError("--normalized requires --mean-path and --std-path")
            mean = np.load(args.mean_path).astype(np.float32)
            std = np.load(args.std_path).astype(np.float32)
            data = data.astype(np.float32) * (std + 1e-8) + mean
        joints = recover_from_ric(data.astype(np.float32), joints_num=22)

    print(f"Motion: {joints.shape[0]} frames, {joints.shape[1]} joints")

    # ── load keyframe indices ─────────────────────────────────────────────────
    keyframe_indices: Optional[list[int]] = None
    if args.keyframe_indices:
        src = args.keyframe_indices.strip()
        if src.endswith(".npy"):
            keyframe_indices = np.load(src).astype(int).tolist()
        else:
            keyframe_indices = [int(x) for x in src.split(",") if x.strip()]
        print(f"Keyframes: {keyframe_indices}")

    # ── figure size ───────────────────────────────────────────────────────────
    try:
        w, h = [float(x) for x in args.figsize.lower().split("x")]
    except ValueError:
        raise ValueError("--figsize must be WxH, e.g. 14x5")

    # ── render ────────────────────────────────────────────────────────────────
    fig = render_ghost_trail(
        joints=joints,
        keyframe_indices=keyframe_indices,
        n_ghosts=args.n_ghosts,
        stride=args.stride,
        spread=args.spread,
        elev=args.elev,
        azim=args.azim,
        figsize=(w, h),
        dpi=args.dpi,
        bone_lw=args.bone_lw,
        joint_size=args.joint_size,
        regular_color=args.regular_color,
        keyframe_color=args.keyframe_color,
        show_floor=not args.no_floor,
        floor_tiles=args.floor_tiles,
        show_axes=args.show_axes,
        title=args.title,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = args.out
    if not out_path:
        base = os.path.splitext(args.motion_file)[0]
        out_path = f"{base}_ghost.png"

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
