"""Scarlet & Violet ACE SPEC Trainer frame assets, built once from real prints and cached.

template(kind)   that kind's (Item, Tool, Stadium) ACE SPEC card with the name, effect text and footer taken off:
                 RGBA, alpha = how much of the frame shows (0 in the art window, the print's own soft rim at its
                 edge); the ACE SPEC banners, the rule pill and the kind's tab are kept

The pink frame is shaded a little differently in Temporal Forces and Twilight Masquerade; the five later sets print
one frame, pixel for pixel in the TCG Live extract, and the pool is those. Every Scarlet & Violet ACE SPEC Trainer is
an Item, a Pokemon Tool or a Stadium: there is no ACE SPEC Supporter. The name, text and illustrator are white, so
the ink of a print is its white pixels (as ace.py finds it); the footer and copyright line are found as the other
Trainers' (trainerframe).
"""
import numpy as np

from . import trainerframe as tf
from .imaging import dilate

KINDS = ("Item", "Tool", "Stadium")

# Shrouded Fable, Stellar Crown, Surging Sparks and Prismatic Evolutions.
POOL = {
    "Item": ["sv06.5-058", "sv06.5-062", "sv08.5-116", "sv08.5-119", "sv08.5-128", "sv08.5-131", "sv08-164",
             "sv08-176", "sv08-182", "sv08-183", "sv08-185", "sv08-186"],
    "Tool": ["sv08.5-117", "sv08.5-129", "sv07-134", "sv07-142", "sv08-162"],
    "Stadium": ["sv06.5-060", "sv07-136"],
}

NAME_ZONE = (40, 78, 690, 136)
TEXT_ZONE = (80, 548, 656, 884)
ZONES = (NAME_ZONE, TEXT_ZONE, tf.FOOTER, tf.COPYRIGHT)
WHITE_CORE = 200           # the least min(R, G, B) of white print (the bar's clean gradient and the pink stay under 90)
INK_GROW = 4               # the glyphs' soft edge


def _ink(rgb):
    """What a print has set on its frame, grown a little: white words in the name bar and panel, the footer's and the
    copyright line's own (trainerframe._row_ink)."""
    mask = np.zeros((tf.H, tf.W), bool)
    for x0, y0, x1, y1 in (NAME_ZONE, TEXT_ZONE):
        mask[y0:y1, x0:x1] = np.clip(rgb[y0:y1, x0:x1], 0, 255).min(axis=2) >= WHITE_CORE
    mask = dilate(mask, INK_GROW)
    mask |= dilate(tf._row_ink(rgb, tf.FOOTER) | tf._row_ink(rgb, tf.COPYRIGHT), tf.INK_GROW)
    return mask


def template(kind):
    """RGBA float32 733x1024: the kind's ACE SPEC frame with nothing printed on it; alpha 0 in the art window."""
    return tf.build(kind, POOL, ZONES, _ink, f"ace-template-{kind}.png")


def build_all():
    for kind in KINDS:
        template(kind)
