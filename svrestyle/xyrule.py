"""The XY Mega Evolution rule box, set on Gen 9 cards (the editor's "mega_rule").

Every XY and Generations "M ... EX" card prints the same box under its art window: a silver plate with the Mega
Evolution symbol, a dark "Mega Evolution rule" tab and a gold panel reading "When 1 of your Pokémon becomes a Mega
Evolution Pokémon, your turn ends." (TCGdex carries no rule text: read off xy2-13 and the SOURCES prints). It is the
same box on every print, but most prints run their art over its right end, so it is lifted as the median of SOURCES,
each moved onto REF's box by its tab (template match), and cut along OUTLINE, measured on that median.

XY hangs the box from the silver bar that closes the art window; Gen 9 cards close theirs with the Pokedex strip
(regular) or the window bar (ex), so the box hangs from those, unscaled, and abilities and attacks start under it.
"""
import cv2
import numpy as np

from . import cache, layers, tcg
from .imaging import load_rgba, save_rgba

W, H = 733, 1024
VERSION = 1
KIND = "mega_rule"                 # the layers kind the box is recorded under (foil plates cut it out)
REF = "xy2-13"                     # M Charizard-EX: the box every source is moved onto
# XY / Generations "M ... EX" prints (TCGdex stage MEGA) whose tab matches REF's at TAB_MATCH or better. Left out:
# Evolutions (xy12) and the Primal Reversion cards, which print other boxes, and reprints of the same art (the "a"
# numbers), which would weigh their art twice.
SOURCES = (
    "xy2-13", "g1-12", "g1-18", "g1-RC31", "xy1-2", "xy1-30", "xy10-118", "xy10-121", "xy10-26", "xy10-69",
    "xy11-109", "xy11-112", "xy11-79", "xy2-107", "xy2-108", "xy2-109", "xy2-69", "xy2-79", "xy3-112", "xy3-113",
    "xy3-5", "xy3-55", "xy4-120", "xy4-121", "xy4-24", "xy4-35", "xy5-106", "xy5-156", "xy5-94", "xy6-102",
    "xy6-105", "xy6-59", "xy6-61", "xy7-28", "xy7-43", "xy7-8", "xy7-85", "xy7-88", "xy7-92", "xy7-98", "xy8-154",
    "xy8-156", "xy8-159", "xy8-160", "xy8-35", "xy8-64", "xy9-115", "xy9-120", "xy9-27", "xy9-77", "xyp-XY104",
    "xyp-XY158", "xyp-XY198", "xyp-XY200", "xyp-XY35", "xyp-XY44", "xyp-XY86", "xyp-XY87", "xyp-XY98",
)
TAB = (120, 612, 400, 642)         # x0, y0, x1, y1 on REF: the dark "Mega Evolution rule" tab and its lettering
TAB_REACH = 40                     # px up or down a print's box may sit from REF's (xy6-61, xy7-43: 22-23 lower)
TAB_MATCH = 0.95                   # a print whose tab matches worse has art over it (or another box)
MIN_SOURCES = 20                   # fewer matching prints leave art in the median
CROP = (36, 598, 694, 693)         # x0, y0, x1, y1 on REF: the box with a few pixels round it
# The box's outline on REF (x, y; pixel edges): straight top and bottom, the right end slanting with the gold panel,
# the left end slanting in to a short upright side that rounds into the bottom. Measured on the median of SOURCES:
# the bottom where its dark rim ends (689.5), the left end where the prints agree, the right end on its edge's
# gradient (687 at y 604, 638 at 666, 623 at 686), a pixel in so its soft edge takes no art.
OUTLINE = ((65.0, 601.5), (688.0, 601.5), (619.5, 689.5), (54.0, 689.5), (49.5, 687.0), (46.0, 683.0),
           (44.0, 679.0), (42.0, 676.0), (42.0, 668.0))
TOP, BOTTOM = 601.5, 689.5         # the box's top and bottom edges on REF
# The dark rim along the box's top, rows y0..y1, is one shade along each row. Right of RIM_FROM art runs over it on
# about half the prints, which tints the median: it takes each row's median over RIM_CLEAN (x0, x1), above the tab.
RIM, RIM_CLEAN, RIM_FROM = (602, 612), (300, 420), 420
OUTLINE_SS = 4                     # supersampling of the outline's anti-aliased edge
# Gen 9: the lower edge of the strip that closes the art window, where the box's top goes. Regular: the Pokedex
# strip, rows 484-511 (me01-056); ex: the window bar, rows 488-510 (me01-077, me01-003, me02-013).
HANG = {"regular": 511.5, "ex": 510.5}
GAP = 20                           # px from the box's bottom to the first attack's cost icons (xy2-13, xy3-55: 710)


def _asset():
    return cache.path("assets", f"xyrule-v{VERSION}.png")


def _offset(gray, tab):
    """(dy, score): how far a print's box sits below REF's, by its tab."""
    x0, y0, x1, y1 = TAB
    window = gray[y0 - TAB_REACH:y1 + TAB_REACH, x0:x1]
    match = cv2.matchTemplate(window, tab, cv2.TM_CCOEFF_NORMED)
    _, score, _, (_, my) = cv2.minMaxLoc(match)
    return my - TAB_REACH, score


def _gray(rgba):
    return cv2.cvtColor(np.clip(rgba[..., :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _inside(x, y, polygon):
    """bool grid: which points (x across, y down) lie inside the polygon (even-odd rule)."""
    out = np.zeros((len(y), len(x)), bool)
    x, y = x[None, :], y[:, None]
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        out ^= crosses & (x < ax + (y - ay) * (bx - ax) / (by - ay))
    return out


def outline_alpha():
    """HxW float32 0..1 over CROP: the share of each pixel inside OUTLINE (pixel r spans r - 0.5 to r + 0.5)."""
    x0, y0, x1, y1 = CROP
    sub = (np.arange(OUTLINE_SS) + 0.5) / OUTLINE_SS - 0.5
    xs = (np.arange(x0, x1)[:, None] + sub).ravel()
    ys = (np.arange(y0, y1)[:, None] + sub).ravel()
    inside = _inside(xs, ys, OUTLINE)
    return inside.reshape(y1 - y0, OUTLINE_SS, x1 - x0, OUTLINE_SS).mean(axis=(1, 3)).astype(np.float32)


def build():
    """RGBA over CROP: the median of the SOURCES' boxes, alpha the outline."""
    ref = tcg.web_scan(REF)
    x0, y0, x1, y1 = TAB
    tab = _gray(ref)[y0:y1, x0:x1]
    cx0, cy0, cx1, cy1 = CROP
    boxes = []
    for card_id in SOURCES:
        scan = ref if card_id == REF else tcg.web_scan(card_id)
        dy, score = _offset(_gray(scan), tab)
        if score >= TAB_MATCH:
            boxes.append(scan[cy0 + dy:cy1 + dy, cx0:cx1, :3])
    if len(boxes) < MIN_SOURCES:
        raise ValueError(f"only {len(boxes)} of {len(SOURCES)} XY Mega cards show the rule box where {REF} does")
    rgb = np.median(np.stack(boxes), axis=0)
    (r0, r1), (c0, c1) = RIM, RIM_CLEAN
    rows = slice(r0 - cy0, r1 - cy0)
    rgb[rows, RIM_FROM - cx0:] = np.median(rgb[rows, c0 - cx0:c1 - cx0], axis=1, keepdims=True)
    return np.dstack([rgb, outline_alpha() * 255]).astype(np.float32)


def sprite():
    """RGBA float32 over CROP (built once, cached)."""
    out = _asset()
    if out.exists():
        return load_rgba(out)
    box = build()
    save_rgba(box, out)
    return box


def shift(family):
    """Whole px the box moves from REF to a Gen 9 card of family ("regular" | "ex"): it is not resampled."""
    return round(HANG[family] - TOP)


def draw(canvas, family):
    """Set the box on canvas (RGBA float32 733x1024) under the strip that closes a card of family's art window."""
    box = sprite()
    x0, y0 = CROP[0], CROP[1] + shift(family)
    a = box[..., 3:4] / 255
    region = canvas[y0:y0 + box.shape[0], x0:x0 + box.shape[1]]
    region[..., :3] = region[..., :3] * (1 - a) + box[..., :3] * a
    with layers.kind(KIND):
        layers.record(a[..., 0], x0, y0)


def floor(family):
    """y the first ability badge or attack's cost icons may start at on a card of family with the box."""
    return BOTTOM + shift(family) + GAP
