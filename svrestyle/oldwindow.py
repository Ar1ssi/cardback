"""Where the art window is on a Trainer card older than HeartGold SoulSilver (or printed in such a layout again:
XY Evolutions).

Those scans are photographs of printed cards: each sits a few px off another of its layout and up to a few percent
larger or smaller, so no fixed window fits them all, and the art's own edges make the window's edges hard to tell
from the frame's. Instead the card's frame is registered to its layout's: along each axis the card's edge profile
(the mean gradient along the other axis, so the frame's long lines stand out and the art and text average away) is
matched to the layout's mean profile by a shift and a scale, and the layout's art window, measured on its
registered prints, is carried back onto the card.

A layout's mean profile is built once from the prints in its POOL (registered to one another) and cached.
"""
import warnings

import cv2
import numpy as np

from . import cache, tcg

W, H = 733, 1024
ASSET_VERSION = "1"

# layout -> TCGdex set ids that print it (and single cards of other sets), prints its mean profile is built from (all
# regular Trainers; the first sets the coordinates), and the art window in those coordinates (x0, y0, x1, y1). The
# window is measured where the registered prints' art varies; on e-Card prints, whose window has round corners on the
# left, it is the largest rectangle of the Gen 9 window's shape inside.
LAYOUTS = {
    "base": {   # Base Set to Legendary Collection, Wizards promos: yellow TRAINER bar, grey frame
        "sets": ("base1", "base2", "base3", "base4", "base5", "gym1", "gym2", "neo1", "neo2", "neo3", "neo4", "lc",
                 "si1", "basep", "bog", "wp", "sp"),
        "pool": ("base1-70", "base1-82", "base2-64", "base3-60", "base4-107", "base5-76", "gym1-104", "gym2-111",
                 "neo1-94", "neo2-75", "neo4-98", "lc-104"),
        "window": (67, 238, 671, 568),
    },
    "ecard": {  # Expedition, Aquapolis, Skyridge
        "sets": ("ecard1", "ecard2", "ecard3"),
        "pool": ("ecard1-137", "ecard1-141", "ecard1-143", "ecard2-121", "ecard2-133", "ecard2-139", "ecard3-121",
                 "ecard3-129", "ecard3-131"),
        "window": (137, 181, 658, 505),
    },
    "ex": {     # EX Ruby & Sapphire to EX Power Keepers, POP 1-5, Nintendo promos
        "sets": tuple(f"ex{n}" for n in range(1, 17)) + ("ex5.5", "exu", "pop1", "pop2", "pop3", "pop4", "pop5",
                                                          "np"),
        "pool": ("ex1-80", "ex1-90", "ex3-82", "ex6-93", "ex8-84", "ex8-88", "ex10-85", "ex10-94", "ex12-79",
                 "ex14-81", "ex16-79", "pop5-7"),
        "window": (72, 149, 663, 499),
    },
    "dp": {     # Diamond & Pearl to Arceus, POP 6-9, Diamond & Pearl promos
        "sets": ("dp1", "dp2", "dp3", "dp4", "dp5", "dp6", "dp7", "pl1", "pl2", "pl3", "pl4", "pop6", "pop7",
                 "pop8", "pop9", "dpp", "ru1"),
        "pool": ("dp1-105", "dp1-111", "dp3-120", "dp5-81", "dp5-85", "dp7-82", "dp7-85", "pl1-107", "pl1-113",
                 "pl2-92", "pl4-88"),
        "window": (71, 152, 664, 514),
    },
    "evolutions": {     # XY Evolutions and its Pokemon League Gym Badges: Base Set's layout redrawn
        "sets": ("xy12",),
        "cards": tuple(f"xyp-XY{n}" for n in range(203, 211)),
        "pool": ("xy12-74", "xy12-76", "xy12-77", "xy12-78", "xy12-79", "xy12-80", "xy12-82", "xy12-83", "xy12-85",
                 "xy12-88"),
        "window": (72, 241, 662, 571),
    },
}
# px of each profile matched: the whole width; down to the text panel's top (its text differs card to card)
ACROSS_SPAN, DOWN_SPAN = (10, 723), (10, 640)
SHIFTS = np.arange(-40, 40.1, 1.0)                 # px, coarse search; then refined around the best
SCALES = np.arange(0.93, 1.0701, 0.005)
FINE_SHIFTS, FINE_SCALES = np.arange(-1.0, 1.01, 0.25), np.arange(-0.005, 0.0051, 0.001)
EDGE = 20                  # px of the scan's rim left out of the profiles: the card's cut edge and its shadow
POOL_ROUNDS = 3            # rounds of registering the pool to its own mean profile
MIN_MATCH = 0.4            # profile correlation below which the card is likely not this layout
INSET = 2                  # px kept inside the carried window: registration is good to about 1 px


def _asset(name):
    return cache.path("assets", f"trainer-v{ASSET_VERSION}", name)


def layout_of(card_id):
    set_id = card_id.split("-", 1)[0]
    return next((name for name, spec in LAYOUTS.items() if set_id in spec["sets"] or card_id in spec.get("cards", ())),
                None)


def profiles(rgb):
    """(across, down): mean |d/dx| down the card's rows, mean |d/dy| along its columns, the scan's rim left out."""
    grey = cv2.cvtColor(np.clip(rgb[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    across = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))[EDGE:H - EDGE].mean(0)
    down = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))[:, EDGE:W - EDGE].mean(1)
    return across, down


def _mapped(profile, shift, scale, centre):
    """profile sampled at the layout coordinates: card position = centre + scale * (layout - centre) + shift."""
    xs = np.arange(len(profile), dtype=np.float32)
    return np.interp(centre + scale * (xs - centre) + shift, xs, profile)


def _correlation(profile, ref, span, shift, scale, centre):
    lo, hi = span
    p, r = _mapped(profile, shift, scale, centre)[lo:hi], ref[lo:hi]
    p, r = p - p.mean(), r - r.mean()
    return float(p @ r) / (float(np.linalg.norm(p) * np.linalg.norm(r)) + 1e-6)


def fit(profile, ref, span):
    """-> (shift, scale, correlation) mapping layout coordinates onto the card along one axis."""
    centre = len(profile) / 2
    best = max((_correlation(profile, ref, span, d, s, centre), d, s) for s in SCALES for d in SHIFTS)
    _, d0, s0 = best
    best = max((_correlation(profile, ref, span, d0 + d, s0 + s, centre), d0 + d, s0 + s)
               for s in FINE_SCALES for d in FINE_SHIFTS)
    corr, d, s = best
    return float(d), float(s), corr


def reference(layout):
    """The layout's mean (across, down) profiles in its own coordinates; built from its POOL once, cached."""
    out = _asset(f"layout-{layout}.npz")
    if out.exists():
        with np.load(out) as f:
            return f["across"], f["down"]
    pool = [profiles(tcg.scan(i)) for i in LAYOUTS[layout]["pool"]]
    across, down = pool[0]
    for _ in range(POOL_ROUNDS):
        fits = [(fit(a, across, ACROSS_SPAN), fit(d, down, DOWN_SPAN)) for a, d in pool]
        across = np.mean([_mapped(a, fa[0], fa[1], W / 2) for (a, _), (fa, _) in zip(pool, fits)], 0)
        down = np.mean([_mapped(d, fd[0], fd[1], H / 2) for (_, d), (_, fd) in zip(pool, fits)], 0)
    np.savez(out, across=across, down=down)
    return across, down


def register(rgb, layout):
    """-> ((x shift, x scale, x correlation), (y shift, y scale, y correlation)) of the card against the layout."""
    across, down = profiles(rgb)
    ref_across, ref_down = reference(layout)
    return fit(across, ref_across, ACROSS_SPAN), fit(down, ref_down, DOWN_SPAN)


def window(card_id, rgb):
    """The art window (x0, y0, x1, y1) on this older card's scan. Raises ValueError for a set of no known layout."""
    layout = layout_of(card_id)
    if layout is None:
        raise ValueError(f"{card_id}: no known Trainer layout for set {card_id.split('-', 1)[0]!r}")
    (dx, sx, cx), (dy, sy, cy) = register(rgb, layout)
    if min(cx, cy) < MIN_MATCH:
        warnings.warn(f"{card_id}: its frame matches the {layout} layout poorly ({cx:.2f}, {cy:.2f}); check the art")
    x0, y0, x1, y1 = LAYOUTS[layout]["window"]
    to_x = lambda x: W / 2 + sx * (x - W / 2) + dx
    to_y = lambda y: H / 2 + sy * (y - H / 2) + dy
    return (int(np.ceil(to_x(x0))) + INSET, int(np.ceil(to_y(y0))) + INSET,
            int(np.floor(to_x(x1))) - INSET, int(np.floor(to_y(y1))) - INSET)
