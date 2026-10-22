"""Mega Evolution regular-card frame assets, built once from real Mega Evolution (me1) scans and cached.

template(type)    that type's Basic card with every printed word, number and icon inpainted away (LaMa); the
                  art window still holds the reference card's art, which the compositor covers
stage_sprite(n)   the silver Stage 1 / Stage 2 pieces (tab, bar swoosh, evolution portrait ring, "Evolves from"
                  strip), lifted from Kadabra and Alakazam; type-independent

Positions are at 733x1024, measured on the pokemontcg.io me1 scans.
"""
import functools

import cv2
import numpy as np

from . import cache, inpaint, tcg
from .imaging import dilate, disc_sprite, drop_small, fill_holes, hsv, load_rgba, save_rgba

W, H = 733, 1024
ASSET_VERSION = "3"

# Regular (non-ex) cards per type from Mega Evolution (me01) and Phantasmal Flames (me02). Cards of one type share
# the same bar, panel and footer texture, so a pixel hidden by text on one card is taken from another. The first
# card is the base; Basics come first because only they share the name bar (Stage cards add silver pieces there).
POOL = {
    "Grass": ["me01-001", "me01-004", "me01-006", "me01-008", "me01-011", "me01-013", "me01-016", "me02-001",
              "me01-002", "me01-005", "me01-007", "me01-014"],
    "Fire": ["me01-019", "me01-021", "me01-023", "me01-026", "me01-029", "me02-011", "me02-015", "me01-020",
             "me01-024", "me01-027", "me01-030"],
    "Water": ["me01-039", "me01-032", "me01-033", "me01-037", "me01-042", "me01-044", "me02-021", "me01-038",
              "me01-040", "me01-041", "me01-043"],
    "Lightning": ["me01-045", "me01-049", "me01-051", "me01-052", "me02-030", "me02-032", "me01-046", "me01-047",
                  "me01-053"],
    "Psychic": ["me01-054", "me01-057", "me01-058", "me01-062", "me01-065", "me01-067", "me01-055", "me01-056",
                "me01-059", "me01-063"],
    "Fighting": ["me01-068", "me01-072", "me01-074", "me01-076", "me01-078", "me01-082", "me01-069", "me01-073",
                 "me01-079", "me01-083"],
    "Darkness": ["me01-089", "me01-087", "me01-091", "me02-054", "me02-057", "me02-059", "me02-060", "me01-090",
                 "me01-092", "me01-085", "me02-055"],
    "Metal": ["me01-096", "me01-095", "me02-071", "me02-073", "me02-074", "me01-097", "me01-098", "me01-099",
              "me02-072"],
    "Dragon": ["me01-101"],                                   # the only regular Dragon in me01/me02
    "Colorless": ["me01-102", "me01-105", "me01-106", "me01-107", "me01-109", "me02-076", "me02-078",
                  "me01-103", "me01-108", "me01-110"],
}
TYPES = {t: ids[0] for t, ids in POOL.items()}
BASIC_PSYCHIC, STAGE_REFS = "me01-054", {1: "me01-055", 2: "me01-056"}    # Abra; Kadabra, Alakazam

ART = (59, 100, 674, 484)                 # x0, y0, x1, y1 of the art window (left/top rim just outside)
NAME_ZONE = (136, 24, 0, 94)              # x0, y0, (x1 = type icon's left edge), y1
DEX_STRIP = (96, 486, 652, 508)
PANEL = (44, 512, 690, 870)
ROW_VALUES = [(112, 880, 202, 910), (302, 880, 404, 910), (515, 880, 700, 910)]   # weakness/resistance/retreat
FOOTER = (30, 914, 703, 995)
# The weakness row's lower rule (silver, its shadow fading into the panel under it) and the panel's bottom rim: rows
# y0, y1. They lie under every card's grown print mask, so the fills blur them; they print the same on every card of
# a type, so template() takes them from the median of the type's prints.
LOWER_RULE = (911, 922)
PRINT_HOLE = 3000                     # px: biggest hole in print that is print (a cost icon's middle is ~1100)
SHAPE_MIN, SPECK_NEAR = 16, 6         # px: smaller standouts are grain unless this near a bigger one
PANEL_RIM = (989, 998)
COPYRIGHT = (120, 998, 620, 1021)
# Where each band's fill may take its colour from (row between its lines, panel above the border, border).
CONTEXT = {ROW_VALUES[0]: (30, 880, 703, 910), FOOTER: (30, 914, 703, 995), COPYRIGHT: (0, 998, W, 1024)}
ICON_ZONE = (620, 18, 712, 104)           # name-bar type icon search box
# Mega Evolution energy icons, used on every restyled card. The name-bar type icon (bevelled) sits at the same place
# on every me1 Basic: centre x, y and radius of its outer rim, measured on all ten types. Attack costs, weakness,
# resistance and retreat use the flatter cost icon, lifted from the first cost slot of a me1 attack; Dragon never
# appears in a cost, so it uses its name-bar icon everywhere.
ICON = (669.2, 60.8, 26.3)
COST_ICONS = {
    "Grass": ("me01-001", 70.92, 653.66, 18.78), "Colorless": ("me01-004", 70.98, 653.65, 18.46),
    "Fire": ("me01-019", 71.03, 728.25, 18.70), "Water": ("me01-039", 70.89, 653.49, 18.86),
    "Lightning": ("me01-045", 70.94, 673.98, 18.39), "Psychic": ("me01-054", 70.79, 653.65, 18.55),
    "Fighting": ("me01-068", 70.79, 728.14, 18.70), "Darkness": ("me01-089", 70.90, 673.98, 18.19),
    "Metal": ("me01-096", 70.92, 674.00, 18.44),
}
ICON_TRIM = 0.2          # px trimmed off the rim, which is blended with the card behind it
NAME_RIGHT_LIMIT = 520                    # names stay left of the bar's two-tone wave / Stage swoosh
# Stage pieces: the tab, the bar swoosh, and the portrait ring + "Evolves from" strip (Kadabra/Alakazam).
STAGE_TAB = (0, 14, 142, 68)
STAGE_SWOOSH = (440, 18, 600, 100)
STAGE_PORTRAIT = (0, 60, 520, 190)
EVOLVES_STRIP = (128, 92, 470, 122)
PORTRAIT_DISC = (73.0, 128.0, 47.0)       # light disc inside the evolution ring: centre x, y, radius      # x0, y0, x1, y1 searched for strip rows


def _asset(name):
    return cache.path("assets", f"me-v{ASSET_VERSION}", name)


def ink(rgb, box, threshold=38, k=41):
    """Pixels in box that stand out from the local background (text, icons, boxes), as a full-size mask."""
    x0, y0, x1, y1 = box
    pad = k
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = np.clip(rgb[Y0:Y1, X0:X1, :3], 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(crop, k).astype(np.float32)
    diff = np.abs(crop.astype(np.float32) - bg).max(-1) > threshold
    mask = np.zeros((H, W), bool)
    mask[Y0:Y1, X0:X1] = diff
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = True
    return mask & keep


def type_icon(rgb):
    """(cx, cy, r) of the name-bar type icon."""
    x0, y0, x1, y1 = ICON_ZONE
    grey = cv2.cvtColor(np.clip(rgb[y0:y1, x0:x1, :3], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(cv2.GaussianBlur(grey, (5, 5), 1.2), cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=80, param2=18, minRadius=19, maxRadius=27)
    if circles is None:
        raise ValueError("type icon not found in the name bar")
    cx, cy, r = circles[0][0]
    return x0 + float(cx), y0 + float(cy), float(r)


def _printed(rgb, basic):
    """Mask of everything printed per card (words, numbers, energy icons, set boxes), grown a little. On a Stage
    card the whole name zone counts as printed: its bar carries the stage pieces."""
    cx, cy, r = type_icon(rgb)
    x0, y0, _, y1 = NAME_ZONE
    mask = np.zeros((H, W), bool)
    if basic:
        mask |= ink(rgb, (x0, y0, int(cx - r - 3), y1), threshold=45)
    else:
        mask[y0:y1, x0:int(cx - r - 3)] = True
        mask[:200, :140] = True                    # stage tab and evolution portrait
    for box in [DEX_STRIP, PANEL, FOOTER, COPYRIGHT]:
        mask |= ink(rgb, box)
    # Water's and Darkness's grain stands out in specks of a few px all over: grown, they would cover the footer
    # whole, and no card would give it. A speck counts only beside a shape of print (punctuation, an i's dot).
    shapes = drop_small(mask, SHAPE_MIN)
    mask = shapes | (mask & dilate(shapes, SPECK_NEAR))
    for box in ROW_VALUES:
        mask |= row_ink(rgb, box)
    return fill_holes(dilate(mask, 3), max_area=PRINT_HOLE)     # what print encloses: counters, an icon's middle


def row_ink(rgb, box, threshold=60):
    """Ink in the weakness row: against each row's median colour along the band (a run of retreat icons is
    wider than ink()'s window, which would take the icons for the background)."""
    x0, y0, x1, y1 = box
    band = rgb[y0:y1, 40:690, :3]
    bg = np.median(band, axis=1, keepdims=True)
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = (np.abs(rgb[y0:y1, x0:x1, :3] - bg).max(-1) > threshold)
    return mask


def template(type_name):
    """RGBA 733x1024: the type's Basic card with name, HP, dex line, attacks, weakness/resistance/retreat values
    and footer removed; tab, bar, type icon, panel, weakness-row labels and lines kept.

    Each printed pixel of the base card is replaced from the first other card of the type that has no print
    there (same texture, so no seam); LaMa fills only what every card covers."""
    if type_name not in POOL:
        raise ValueError(f"no Mega Evolution frame for type {type_name!r} (have: {', '.join(POOL)})")
    out = _asset(f"template-{type_name}.png")
    if out.exists():
        return load_rgba(out)
    ids = POOL[type_name]
    base = tcg.scan(ids[0])
    clean = base.copy()
    need = _printed(base[..., :3], basic=True)
    need[ART[1]:ART[3], ART[0]:ART[2]] = False          # the art window is covered by the new art anyway
    for card_id in ids[1:]:
        if not need.any():
            break
        other = tcg.scan(card_id)
        basic = tcg.card(card_id).get("stage") == "Basic"
        take = need & ~_printed(other[..., :3], basic)
        clean[take, :3] = other[take, :3]
        need &= ~take
    # The dex strip is a horizontal silver bar: what is still covered there takes each row's median.
    sx0, sy0, sx1, sy1 = DEX_STRIP
    for y in range(sy0, sy1):
        row = need[y, sx0:sx1]
        if row.any():
            free = ~row[60:-20]
            if free.any():
                clean[y, sx0:sx1][row, :3] = np.median(clean[y, sx0 + 60:sx1 - 20][free, :3], axis=0)
                need[y, sx0:sx1] = False
    if need.any():
        filled = inpaint.inpaint(clean, need)
        # LaMa copies neighbouring letters into the thin weakness row and the footer. Those bands only shade
        # slowly, so their holes take the colour of the surrounding clean pixels (normalized blur, wide along x,
        # narrow along y so the row's lines stay put) plus a little of the band's own grain.
        labels = dilate(ink(base[..., :3], (36, ROW_VALUES[0][1], 702, ROW_VALUES[0][3])), 2) & ~need
        known = (~need & ~labels).astype(np.float32)
        grain = clean[..., :3] - cv2.GaussianBlur(clean[..., :3], (0, 0), 1.2)
        noise = np.roll(grain * known[..., None], 90, axis=1)           # clean grain from further along the row
        for boxes, ctx, sx, sy in ((ROW_VALUES, CONTEXT[ROW_VALUES[0]], 30, 1.5), ([FOOTER], CONTEXT[FOOTER], 14, 5),
                                   ([COPYRIGHT], CONTEXT[COPYRIGHT], 14, 5)):
            band = np.zeros((H, W), bool)
            for x0, y0, x1, y1 in boxes:
                band[y0:y1, x0:x1] = True
            fill = need & band
            inside = np.zeros((H, W), np.float32)
            cx0, cy0, cx1, cy1 = ctx
            inside[cy0:cy1, cx0:cx1] = 1
            k = known * inside
            num = cv2.GaussianBlur(clean[..., :3] * k[..., None], (0, 0), sigmaX=sx, sigmaY=sy)
            den = cv2.GaussianBlur(k, (0, 0), sigmaX=sx, sigmaY=sy)[..., None]
            colour = np.where(den > 1e-3, num / np.maximum(den, 1e-3), filled[..., :3])
            feather = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 1.5)[..., None] * fill[..., None]
            filled[..., :3] = filled[..., :3] * (1 - feather) + (colour + 0.7 * noise) * feather
        clean = filled
    for rows in (LOWER_RULE, PANEL_RIM):
        clean[rows[0]:rows[1], :, :3] = _median_rows(ids, rows)
    save_rgba(clean, out)
    return clean


def _median_rows(ids, rows):
    """Rows y0..y1 of the given cards' prints, each pixel their median: what one card prints over there, the
    others do not."""
    y0, y1 = rows
    return np.median(np.stack([tcg.scan(i)[y0:y1, :, :3] for i in ids]), axis=0)


def stage_sprite(stage):
    """RGBA 733x1024, alpha = the silver stage pieces only (tab, bar swoosh, portrait ring with its light disc,
    blank "Evolves from" strip)."""
    out = _asset(f"stage{stage}.png")
    if out.exists():
        return load_rgba(out)
    card = tcg.scan(STAGE_REFS[stage])
    other = tcg.scan(STAGE_REFS[3 - stage])
    basic = tcg.scan(BASIC_PSYCHIC)
    rgb = card[..., :3]
    _, sat, val = hsv(rgb)
    silver = (sat < 0.16) & (val > 0.25)
    differs = np.abs(rgb - basic[..., :3]).max(-1) > 30

    alpha = np.zeros((H, W), bool)
    x0, y0, x1, y1 = STAGE_TAB
    alpha[y0:y1, x0:x1] = (silver | (val < 0.2))[y0:y1, x0:x1]       # the tab's black letters belong to it
    x0, y0, x1, y1 = STAGE_SWOOSH
    alpha[y0:y1, x0:x1] |= (silver & differs)[y0:y1, x0:x1]
    # Ring and strip: silver, the same on Kadabra and Alakazam (their art differs), and absent on Abra.
    x0, y0, x1, y1 = STAGE_PORTRAIT
    same = np.abs(rgb - other[..., :3]).max(-1) < 24
    piece = np.zeros((H, W), bool)
    piece[y0:y1, x0:x1] = (silver & same & differs)[y0:y1, x0:x1]
    piece = cv2.morphologyEx(piece.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    piece = fill_holes(piece)
    x0, y0, x1, y1 = EVOLVES_STRIP
    for y in range(y0, y1):                        # the strip's letters break it open: close each strip row
        cols = np.nonzero(piece[y, x0:x1])[0]
        if len(cols) > 20:
            piece[y, x0 + cols[0]:x0 + cols[-1] + 1] = True
    alpha |= piece
    alpha = cv2.morphologyEx(alpha.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    sprite = card.copy()
    # Blank strip: rows of the strip right of the ring take their median (drops "Evolves from ...").
    x0, y0, x1, y1 = EVOLVES_STRIP
    for y in range(y0, y1):
        cols = np.nonzero(piece[y, x0:x1])[0]
        if len(cols) > 20:
            sprite[y, x0 + cols, :3] = np.median(rgb[y, x0 + cols[len(cols) // 2:]], axis=0)   # right half: no text
    # Empty the portrait disc: Kadabra's Abra is replaced by the light disc's own shading (the plain pixels,
    # spread over the rest by a normalized blur).
    cx, cy = PORTRAIT_DISC[:2]
    yy, xx = np.mgrid[:H, :W]
    disc = np.hypot(xx - cx, yy - cy) < PORTRAIT_DISC[2]
    plain = disc & (sat < 0.08) & (val > 0.75)
    w = plain.astype(np.float32)
    num = cv2.GaussianBlur(rgb * w[..., None], (0, 0), 9)
    den = cv2.GaussianBlur(w, (0, 0), 9)[..., None]
    fill = disc & ~plain
    sprite[fill, :3] = (num / np.maximum(den, 1e-3))[fill]
    sprite[..., 3] = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0.6) * 255
    save_rgba(sprite, out)
    return sprite


@functools.lru_cache(maxsize=None)
def icon(type_name, kind="cost"):
    """(sprite, radius): the Mega Evolution energy icon of a type ('cost' or 'name' style; see imaging.disc_sprite)
    and its radius as measured on the scan, to scale from."""
    if type_name not in TYPES:
        raise ValueError(f"no Mega Evolution energy icon for type {type_name!r}")
    if kind == "cost" and type_name in COST_ICONS:
        card_id, cx, cy, r = COST_ICONS[type_name]
    else:
        card_id, (cx, cy, r) = TYPES[type_name], ICON
    return disc_sprite(tcg.scan(card_id), cx, cy, r - ICON_TRIM), r


def build_all():
    for t in TYPES:
        template(t)
    for s in STAGE_REFS:
        stage_sprite(s)
