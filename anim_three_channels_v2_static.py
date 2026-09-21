"""
realqm/figures/animations/anim_three_channels_v2_static.py

A static, publication-quality frame of the refraction geometry
through a tilted glass slab.

Shows:
  - The slab, tilted 25 degrees counter-clockwise.
  - The incoming beam, horizontal, entering the slab's upper face.
  - The bend at the entry point (Snell: theta_1 -> theta_2).
  - The internal beam, traveling at the internal angle.
  - The bend at the exit point (Snell: theta_2 -> theta_1).
  - The outgoing beam, horizontal, displaced below the incoming.
  - The reflected branch at the entry point.
  - Labels for the angles, the beams, and the slab.

This is the version intended for inclusion in the paper as a
static figure.

Run from Z-3/:
    python -m realqm.figures.animations.anim_three_channels_v2_static

Outputs:
    realqm/figures/animations/output/anim_v2_static.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from realqm.parameters.constants import GREEN_PHOTON
from realqm.slab.slab import default_slab


# ---------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------

CANVAS_W = 12.8
CANVAS_H = 7.2

SLAB_CENTER = (7.0, 3.6)
SLAB_HALF_LENGTH = 2.2
SLAB_HALF_THICKNESS = 0.9
SLAB_ROTATION_DEG = 25.0

ENTRY_FRACTION = 0.30

# Beam half-widths (canvas units). Thinner than v1.
BEAM_HALF = 0.07
REFLECT_HALF = 0.03

# Reflected branch length (canvas units)
REFLECT_LENGTH = 1.3

# Colors
COLOR_INCOMING = "#1f4e79"
COLOR_REFLECTED = "#c0392b"
COLOR_SLAB_FILL = "#eef2f7"
COLOR_SLAB_EDGE = "#1f4e79"
COLOR_TEXT = "#222222"
COLOR_ANGLE = "#999999"


# ---------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------

def add(u, v): return (u[0] + v[0], u[1] + v[1])
def sub(u, v): return (u[0] - v[0], u[1] - v[1])
def scale(u, s): return (u[0] * s, u[1] * s)
def dot(u, v): return u[0] * v[0] + u[1] * v[1]
def unit(v):
    m = math.hypot(v[0], v[1])
    return (v[0] / m, v[1] / m)


def line_intersection(p1, d1, p2, d2):
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    dp = sub(p2, p1)
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / denom
    point = add(p1, scale(d1, t))
    return (t, s, point)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def compute_geometry(n_bulk):
    phi = math.radians(SLAB_ROTATION_DEG)
    u = (math.cos(phi), math.sin(phi))
    n = (-math.sin(phi), math.cos(phi))
    C = SLAB_CENTER
    a = SLAB_HALF_LENGTH
    b = SLAB_HALF_THICKNESS

    c1 = add(add(C, scale(u, +a)), scale(n, +b))
    c2 = add(add(C, scale(u, +a)), scale(n, -b))
    c3 = add(add(C, scale(u, -a)), scale(n, +b))
    c4 = add(add(C, scale(u, -a)), scale(n, -b))

    entry_p1, entry_p2 = c3, c1
    exit_p1, exit_p2 = c4, c2
    entry_dir = unit(sub(entry_p2, entry_p1))

    entry_point = add(entry_p1, scale(sub(entry_p2, entry_p1), ENTRY_FRACTION))

    d_in = (1.0, 0.0)
    n_entry = n

    cos_theta1 = dot(d_in, scale(n_entry, -1.0))
    cos_theta1 = max(-1.0, min(1.0, cos_theta1))
    theta_1 = math.acos(cos_theta1)

    sin_theta2 = math.sin(theta_1) / n_bulk
    sin_theta2 = max(-1.0, min(1.0, sin_theta2))
    theta_2 = math.asin(sin_theta2)

    t_entry = entry_dir
    if dot(d_in, t_entry) < 0:
        t_entry = scale(t_entry, -1.0)

    d_int = unit(add(
        scale(t_entry, math.sin(theta_2)),
        scale(n_entry, -math.cos(theta_2)),
    ))

    result = line_intersection(entry_point, d_int, exit_p1, sub(exit_p2, exit_p1))
    if result is None:
        raise RuntimeError("Internal ray parallel to exit face.")
    t_int, s_exit, exit_point = result

    d_out = d_in
    d_reflected = unit(sub(d_in, scale(n_entry, 2.0 * dot(d_in, n_entry))))

    return {
        "corners": (c1, c2, c3, c4),
        "entry_point": entry_point,
        "exit_point": exit_point,
        "d_in": d_in,
        "d_int": d_int,
        "d_out": d_out,
        "d_reflected": d_reflected,
        "n_entry": n_entry,
        "t_entry": t_entry,
        "theta_1": theta_1,
        "theta_2": theta_2,
        "internal_length": t_int,
    }


# ---------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------

def draw_thick_line(ax, p0, p1, half_width, color, zorder=3):
    d = unit(sub(p1, p0))
    perp = (-d[1], d[0])
    a = add(p0, scale(perp, +half_width))
    b = add(p1, scale(perp, +half_width))
    c = add(p1, scale(perp, -half_width))
    e = add(p0, scale(perp, -half_width))
    poly = mpatches.Polygon([a, b, c, e], closed=True,
                            facecolor=color, edgecolor="none",
                            zorder=zorder)
    ax.add_patch(poly)


# ---------------------------------------------------------------------
# Draw the frame (shared between static and animated versions)
# ---------------------------------------------------------------------

def draw_frame(ax, geo, n_bulk, beam_progress=1.0,
               reflect_progress=1.0, internal_progress=1.0,
               out_progress=1.0):
    """
    Draw one frame of the picture.

    Parameters
    ----------
    beam_progress : float
        How much of the incoming beam is drawn (0..1).
    reflect_progress : float
        How much of the reflected branch is drawn (0..1).
    internal_progress : float
        How much of the internal beam is drawn (0..1).
    out_progress : float
        How much of the outgoing beam is drawn (0..1).
    """
    ax.clear()
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    c1, c2, c3, c4 = geo["corners"]

    # ---- The slab ----
    slab_poly = mpatches.Polygon(
        [c1, c2, c4, c3], closed=True,
        facecolor=COLOR_SLAB_FILL,
        edgecolor=COLOR_SLAB_EDGE,
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(slab_poly)

    # ---- Incoming beam ----
    in_start = (0.5, geo["entry_point"][1])
    in_end_full = geo["entry_point"]
    in_end = add(in_start, scale(sub(in_end_full, in_start), beam_progress))
    if beam_progress > 0:
        draw_thick_line(ax, in_start, in_end, BEAM_HALF, COLOR_INCOMING, zorder=3)

    # ---- Reflected branch ----
    if reflect_progress > 0:
        ref_end = add(geo["entry_point"],
                      scale(geo["d_reflected"], REFLECT_LENGTH * reflect_progress))
        draw_thick_line(ax, geo["entry_point"], ref_end, REFLECT_HALF,
                        COLOR_REFLECTED, zorder=3)

    # ---- Internal beam ----
    if internal_progress > 0:
        int_end = add(geo["entry_point"],
                      scale(sub(geo["exit_point"], geo["entry_point"]),
                            internal_progress))
        draw_thick_line(ax, geo["entry_point"], int_end, BEAM_HALF,
                        COLOR_INCOMING, zorder=3)

    # ---- Outgoing beam ----
    if out_progress > 0:
        out_end_full = (CANVAS_W - 0.5, geo["exit_point"][1])
        out_end = add(geo["exit_point"],
                      scale(sub(out_end_full, geo["exit_point"]), out_progress))
        draw_thick_line(ax, geo["exit_point"], out_end, BEAM_HALF,
                        COLOR_INCOMING, zorder=3)

    # ---- Normal (dashed) at the entry point ----
    n_vis = 1.8
    normal_end = add(geo["entry_point"], scale(geo["n_entry"], n_vis))
    ax.plot([geo["entry_point"][0], normal_end[0]],
            [geo["entry_point"][1], normal_end[1]],
            color=COLOR_ANGLE, linewidth=0.8, linestyle="--", zorder=2)
    ax.text(normal_end[0] + 0.05, normal_end[1] + 0.05, "normal",
            fontsize=10, color=COLOR_ANGLE, family="serif")

    # ---- Labels ----

    # Incoming beam
    ax.text(1.0, geo["entry_point"][1] + 0.25, "incoming beam",
            fontsize=11, color=COLOR_INCOMING, family="serif")

    # Reflected
    ref_label_pos = add(geo["entry_point"],
                        scale(geo["d_reflected"], REFLECT_LENGTH * 0.75))
    ax.text(ref_label_pos[0] - 0.1, ref_label_pos[1] + 0.35, "reflected",
            fontsize=11, color=COLOR_REFLECTED, family="serif",
            ha="right")

    # Transmitted
    out_label_pos = add(geo["exit_point"], (2.0, 0.35))
    ax.text(out_label_pos[0], out_label_pos[1], "transmitted",
            fontsize=11, color=COLOR_INCOMING, family="serif",
            ha="left", va="bottom")

    # Slab label, placed below-left of the internal beam
    # The slab center is at SLAB_CENTER; the internal beam is above
    # and to the right of it. Place the label below-left.
    ax.text(SLAB_CENTER[0] - 0.9, SLAB_CENTER[1] - 1.3, "glass slab",
            fontsize=11, color=COLOR_SLAB_EDGE, family="serif",
            ha="center", va="center", alpha=0.7)

    # theta_1 label, placed above-left of the entry point, clear of
    # the incoming beam and the normal.
    ax.text(geo["entry_point"][0] - 1.6, geo["entry_point"][1] - 0.35,
            f"θ₁ = {math.degrees(geo['theta_1']):.1f}°",
            fontsize=11, color=COLOR_TEXT, family="serif")

    # theta_2 label, placed below-right of the entry point, clear of
    # the internal beam.
    int_mid = add(geo["entry_point"],
                  scale(sub(geo["exit_point"], geo["entry_point"]), 0.5))
    ax.text(int_mid[0] + 0.3, int_mid[1] - 0.45,
            f"θ₂ = {math.degrees(geo['theta_2']):.1f}°",
            fontsize=11, color=COLOR_TEXT, family="serif")

    # ---- Title and subtitle ----
    ax.text(CANVAS_W / 2, CANVAS_H - 0.3,
            "Refraction through a tilted glass slab",
            fontsize=15, color=COLOR_TEXT, family="serif",
            ha="center", va="top")
    ax.text(CANVAS_W / 2, CANVAS_H - 0.7,
            f"Green photon (551 nm), n = {n_bulk:.3f}",
            fontsize=11, color="#666", family="serif",
            ha="center", va="top")


# ---------------------------------------------------------------------
# Main: static version
# ---------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    omega = GREEN_PHOTON.omega
    slab = default_slab(thickness=1.0e-2)
    n_bulk = slab.bulk_refractive_index(omega).real

    geo = compute_geometry(n_bulk)

    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H), dpi=100)
    draw_frame(ax, geo, n_bulk)
    fig.tight_layout()
    png_path = output_dir / "anim_v2_static.png"
    fig.savefig(png_path, dpi=100)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"  n_bulk = {n_bulk:.6f}")
    print(f"  theta_1 = {math.degrees(geo['theta_1']):.4f} deg")
    print(f"  theta_2 = {math.degrees(geo['theta_2']):.4f} deg")
    print(f"  entry point = ({geo['entry_point'][0]:.4f}, {geo['entry_point'][1]:.4f})")
    print(f"  exit point  = ({geo['exit_point'][0]:.4f}, {geo['exit_point'][1]:.4f})")


if __name__ == "__main__":
    main()