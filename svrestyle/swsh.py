"""Strip a Sword & Shield Basic V card down to its art: frame, name bar, text block, energy icons, weakness row,
rule box and footer are masked and rebuilt by LaMa. The art and any Single/Rapid/Fusion Strike stamp are kept.

The V frame is the same on every card, so its parts are fixed regions. The text block is not: its white letters
with a black keyline are found row by row (see find_text), and the energy cost icons by circle detection.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import inpaint
from .imaging import dilate, hsv, luminance

W, H = 733, 1024
TOP_MASKED = 104          # SWSH name bar and its coloured bottom edge
SHIFT_DOWN = 10           # the SV under-bar reaches y 117 (SWSH bar: 104): move the art down to keep its top
TEXT_ZONE = (40, 500, 700, 868)    # x0, y0, x1, y1 the text block can occupy
BOTTOM_MASKED = 868       # weakness row, rule box, footer, bottom border

# Text rows: white glyph pixels touching a black keyline. A row is text if enough of them line up across it.
GLYPH_WHITE_L, GLYPH_WHITE_S, KEYLINE_L = 190, 0.25, 60
ROW_MIN = 10              # keyed pixels for a row to count
BAND_MIN_H, BAND_MIN_PX = 7, 120
COL_GAP = 40              # gap that splits a band into separate runs (cost icons | name | damage)
RUN_MIN_DENSITY = 2.0     # keyed px per column: text runs measured 2.4-22, art beside text 0-1.9
GLYPH_GROW = 4            # covers the keyline's anti-aliasing and the soft drop shadow
BADGE_MIN_PX, BADGE_X = 1500, (40, 250)    # the red Ability badge: a solid red pill at the left
ABILITY_NAME_MAX_X = 560
ICON_X0, ICON_PITCH, ICON_SLOTS, ICON_TOL = 72.5, 44, 5, 6   # SWSH cost icons sit on fixed slots


@dataclass
class TextBand:
    y0: int
    y1: int
    runs: list = field(default_factory=list)   # (x0, x1)


def frame_mask():
    m = np.zeros((H, W), bool)
    m[:TOP_MASKED] = True
    flourish = Image.new("L", (W, H))
    ImageDraw.Draw(flourish).polygon([(0, 95), (104, 95), (104, 112), (50, 250), (0, 250)], fill=255)
    m |= np.asarray(flourish) > 0          # silver V flourish, top left
    m[99:, :24] = True                      # black side borders
    m[99:, 711:] = True
    m[815:, 625:] = True                    # silver flourish, bottom right
    m[BOTTOM_MASKED:] = True
    return m


def find_badges(red):
    """Bounding boxes (x0, y0, x1, y1) of red Ability badges in the text zone."""
    x0, y0, x1, y1 = TEXT_ZONE
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area >= BADGE_MIN_PX and BADGE_X[0] <= bx + x0 and bx + x0 + bw <= BADGE_X[1] and 18 <= bh <= 45:
            boxes.append((bx + x0, by + y0, bx + x0 + bw, by + y0 + bh))
    return boxes


def find_text(rgb):
    """-> (bands, badge boxes, glyph mask). Bands are rows of keylined white text; the mask is their glyphs."""
    L = luminance(rgb)
    hue, sat, val = hsv(rgb)
    white = (L > GLYPH_WHITE_L) & (sat < GLYPH_WHITE_S)
    dark = L < KEYLINE_L
    keyed = white & dilate(dark, 2)
    x0, y0, x1, y1 = TEXT_ZONE
    zone = np.zeros_like(keyed)
    zone[y0:y1, x0:x1] = True
    keyed &= zone
    rows = keyed.sum(1) >= ROW_MIN
    bands = []
    y = y0
    while y < y1:
        if not rows[y]:
            y += 1
            continue
        start = y
        while y < y1 and (rows[y] or rows[y:y + 4].any()):
            y += 1
        if y - start >= BAND_MIN_H and keyed[start:y].sum() >= BAND_MIN_PX:
            bands.append(TextBand(start, y))
    red_all = ((hue < 15) | (hue > 340)) & (sat > 0.55) & (val > 0.4)
    badges = find_badges(red_all)
    # Red counts as text only in an Ability row (badge + red ability name); elsewhere it is art (fire, pink).
    red = np.zeros_like(red_all)
    for bx0, by0, bx1, by1 in badges:
        red[by0 - 6:by1 + 6, bx0:ABILITY_NAME_MAX_X] = red_all[by0 - 6:by1 + 6, bx0:ABILITY_NAME_MAX_X]
    ink = keyed | red
    glyphs = np.zeros_like(keyed)
    for b in bands:
        cols = np.nonzero(ink[b.y0:b.y1].any(0))[0]
        if not len(cols):
            continue
        start = prev = cols[0]
        for c in list(cols[1:]) + [None]:
            if c is None or c - prev > COL_GAP:
                if ink[b.y0:b.y1, start:prev + 1].sum() / (prev + 1 - start) >= RUN_MIN_DENSITY:
                    b.runs.append((int(start), int(prev) + 1))
                if c is not None:
                    start = c
            if c is not None:
                prev = c
        pad = 3
        for rx0, rx1 in b.runs:
            ys, xs = slice(b.y0 - pad, b.y1 + pad), slice(rx0 - pad, rx1 + pad)
            glyphs[ys, xs] |= white[ys, xs] | dark[ys, xs] | red[ys, xs]
    for bx0, by0, bx1, by1 in badges:            # the badge's silver rim and fill are not glyph pixels
        glyphs[by0 - 5:by1 + 5, bx0 - 9:bx1 + 9] = True
    return bands, badges, dilate(glyphs, GLYPH_GROW)


def find_icons(rgb, bands):
    """Energy cost icons left of the attack names: circles near a text band's rows."""
    gray = cv2.GaussianBlur(cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (5, 5), 1)
    x0, x1 = 35, 260
    y0, y1 = TEXT_ZONE[1], TEXT_ZONE[3]
    found = cv2.HoughCircles(gray[y0:y1, x0:x1], cv2.HOUGH_GRADIENT, dp=1, minDist=34,
                             param1=80, param2=18, minRadius=16, maxRadius=25)
    icons = []
    for cx, cy, r in ([] if found is None else found[0]):
        cx, cy = cx + x0, cy + y0
        slot = round((cx - ICON_X0) / ICON_PITCH)
        on_slot = 0 <= slot < ICON_SLOTS and abs(cx - (ICON_X0 + slot * ICON_PITCH)) <= ICON_TOL
        if on_slot and any(b.y0 - 14 <= cy <= b.y1 + 14 for b in bands):
            icons.append((float(cx), float(cy), float(r)))
    return icons


def text_mask(rgba):
    """Everything to rebuild: frame regions, text glyphs, cost icons. -> (mask, report)."""
    rgb = rgba[..., :3]
    bands, badges, glyphs = find_text(rgb)
    icons = find_icons(rgb, bands)
    mask = frame_mask() | glyphs
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy, r in icons:
        mask |= np.hypot(xx - cx, yy - cy) <= r + 7      # icon, its dark rim and shadow
    report = {"bands": [(b.y0, b.y1, b.runs) for b in bands], "badges": badges,
              "icons": [tuple(round(v, 1) for v in i) for i in icons]}
    return mask, report


def clean(rgba):
    """SWSH V scan (733x1024 RGBA float) -> (art shifted down SHIFT_DOWN px with everything but the art rebuilt,
    the mask used, a report of what was found)."""
    mask, report = text_mask(rgba)
    shifted = np.empty_like(rgba)
    shifted[SHIFT_DOWN:], shifted[:SHIFT_DOWN] = rgba[:-SHIFT_DOWN], rgba[0]
    smask = np.empty_like(mask)
    smask[SHIFT_DOWN:], smask[:TOP_MASKED + SHIFT_DOWN] = mask[:-SHIFT_DOWN], True
    return inpaint.inpaint(shifted, smask), mask, report


def overlay(rgba, mask):
    """The scan with the mask painted green, for checking a card before trusting its result."""
    out = rgba[..., :3].copy()
    out[mask] = out[mask] * 0.35 + np.array([0, 255, 0], np.float32) * 0.65
    return out
