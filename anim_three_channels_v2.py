"""
realqm/figures/animations/anim_three_channels_v2.py

An animated version of the tilted-slab refraction figure.

Uses the same geometry as anim_three_channels_v2_static.py, but
adds a timeline:

  Phase 1 (t = 0.00 .. 0.20): incoming beam appears from the left.
  Phase 2 (t = 0.20 .. 0.30): the bend at the entry point, and the
                              reflected branch appears.
  Phase 3 (t = 0.30 .. 0.65): internal beam travels through the slab.
  Phase 4 (t = 0.65 .. 0.80): the bend at the exit point.
  Phase 5 (t = 0.80 .. 1.00): outgoing beam continues to the right.

The animation loops cleanly: at the end of phase 5, the incoming
beam has travelled all the way to the entry point, the internal beam
has traversed the slab, and the outgoing beam has travelled to the
right. When the loop restarts, the incoming beam begins again from
the left.

Note: because this is a one-shot animated demonstration (not a
looping GIF), the loop restart is slightly abrupt. That is fine for
a YouTube clip, which plays once.

Run from Z-3/:
    python -m realqm.figures.animations.anim_three_channels_v2

Outputs:
    realqm/figures/animations/output/anim_v2.gif
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from realqm.parameters.constants import GREEN_PHOTON
from realqm.slab.slab import default_slab

# Import the geometry and drawing from the static module
from realqm.figures.animations.anim_three_channels_v2_static import (
    compute_geometry,
    draw_frame,
    CANVAS_W,
    CANVAS_H,
)


# ---------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------

DURATION_S = 12.0
FPS = 20
N_FRAMES = int(DURATION_S * FPS)


# ---------------------------------------------------------------------
# Progress functions
# ---------------------------------------------------------------------

def incoming_progress(t):
    """How much of the incoming beam is drawn."""
    if t < 0.20:
        return t / 0.20
    return 1.0


def reflect_progress(t):
    """How much of the reflected branch is drawn."""
    if t < 0.20:
        return 0.0
    if t > 0.35:
        return 1.0
    return (t - 0.20) / 0.15


def internal_progress(t):
    """How much of the internal beam is drawn."""
    if t < 0.30:
        return 0.0
    if t > 0.65:
        return 1.0
    return (t - 0.30) / 0.35


def outgoing_progress(t):
    """How much of the outgoing beam is drawn."""
    if t < 0.65:
        return 0.0
    if t > 0.95:
        return 1.0
    return (t - 0.65) / 0.30


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    omega = GREEN_PHOTON.omega
    slab = default_slab(thickness=1.0e-2)
    n_bulk = slab.bulk_refractive_index(omega).real

    geo = compute_geometry(n_bulk)

    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H), dpi=100)

    def update(frame):
        t = frame / max(1, N_FRAMES - 1)
        draw_frame(
            ax, geo, n_bulk,
            beam_progress=incoming_progress(t),
            reflect_progress=reflect_progress(t),
            internal_progress=internal_progress(t),
            out_progress=outgoing_progress(t),
        )
        return []

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         interval=1000 / FPS, blit=False)

    gif_path = output_dir / "anim_v2.gif"
    writer = PillowWriter(fps=FPS)
    anim.save(str(gif_path), writer=writer, dpi=100)
    plt.close(fig)

    print(f"Wrote {gif_path}")
    print(f"  {N_FRAMES} frames, {DURATION_S}s at {FPS} fps")


if __name__ == "__main__":
    main()