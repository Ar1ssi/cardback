"""HP, attack damage, weakness-row values and energy symbols in effect text, set in the print's own glyphs.

Scarlet & Violet and Mega Evolution print these in a Gill Sans whose figures are proportional (at HP size a 1 is
24 px wide, a 0 32) and shaped unlike any Gill Sans face here: set in a font they miss the print by 2-4 px a digit,
the 1 by 6. So each glyph is lifted off scans instead. Every printed instance on the SOURCES cards is cut out of its
background (how far it stands out from a BACKGROUND px median, in units of its ink), upsampled UP x, aligned to the
others and averaged. Where a glyph goes is fitted on the same samples, right-aligned as printed: its x is its own
anchor less the advances of the glyphs right of it and the few kerns between them; its y a fixed row (HP, the
"HP" label set with it) or a fixed offset from the attack name's baseline (damage).

Measured on 23 cards outside SOURCES (summed squared coverage off the print; the font setting this replaces in
brackets): HP 1029 (14367; the best Gill Sans fit 8734), damage 870 (7635), weakness 157 (303), resistance 50 (250).
Weakness and resistance only ever print x2 and -30: any other value is set in Gill Sans as before.

Energy symbols in effect text print bare, in the text's ink, a little taller than its capitals. They are found on
the print by laying a card's own text out as compose does and matching the words either side of each symbol's gap
to the print; what stands out between them is the symbol. Each is placed from its gap's centre and the line's
baseline, scaled with the text when it shrinks. On 51 symbols held out of SOURCES they miss the print by 1210 (the
disc with the icon's symbol knocked out set before: 6924).
"""
import functools
import json
from typing import NamedTuple

import cv2
import numpy as np

from . import cache, tcg
from .imaging import luminance

W, H = 733, 1024
VERSION = 1
UP = 4                          # glyphs are kept at 4x, the supersampling typeset sets text at
BACKGROUND = 61                 # px: median window giving the panel behind the print
INK_SHARE = 85                  # percentile of how far the print stands out that is its full ink
NOISE = 0.2                     # spread of a card's background near its print (ink units) past which it is skipped
PAD = 3                         # px kept round a glyph's cut-out
ALIGN_ROUNDS = 4
AGREE = 0.8                     # least likeness (normalised correlation) of a sample to its glyph's mean
MIN_PAIR = 3                    # samples a pair of glyphs needs for a kern of its own
KERN_RIDGE, ADVANCE_RIDGE, ADVANCE_PRIOR = 4.0, 0.05, 25 * UP  # the fit's pull towards no kern / a common advance

HP_BOX = (470, 30, 650, 95)     # x0, y0, x1, y1: the HP figures and their label, on every SV / ME card
HP_FIGURE_H = 30                # px: shorter shapes there are the label's letters
LABEL = "HP"
DAMAGE_BOX = (585, 515, 705, 880)
DAMAGE_FIGURE_H = (22, 34)      # px: a damage figure's height (suffixes are shorter, body text too)
DAMAGE_FIGURE_W = 23            # px: wider shapes are icons
DAMAGE_RIGHT = 650              # px: a damage ends past this
NAME_LEFT = 200                 # px: the attack name is looked for from here to just short of the damage
NAME_GLYPH_H = (12, 40)
# Weakness-row values, past their icons (and the icons' shadows): x0, y0, x1, y1
ROW_BOXES = {"weakness": (142, 880, 215, 916), "resistance": (336, 880, 410, 916)}
WORDS = 48                      # px of our words matched either side of a symbol's gap
GAP_HALF = 8                    # px either side of a gap's centre left out of them
LINE_HALF = 16                  # px above and below the line's middle they cover
SEARCH = (14, 40)               # px the print may sit off our layout: across, down (ex blocks stack differently)
MATCH = 0.45                    # least correlation of the words with the print
SYMBOL_BOX = (15, 26, 6)        # px: a symbol lies within this far either side of its gap's centre, above and
                                # below the baseline; shapes reaching out of it are the words

# Cards the glyphs are lifted from: plain panels, dark print, every figure and suffix many times over (held-out
# measurement cards excluded).
SOURCES = {
    "energy": (
        "me01-003", "me01-005", "me01-011", "me01-025", "me01-036", "me01-064", "me01-074", "me01-077", "me01-084",
        "me02-007", "me02-013", "me02-015", "me02-016", "me02-018", "me02-039", "me02-040", "me02.5-010", "me02.5-024",
        "me02.5-040", "me02.5-048", "me02.5-053", "me02.5-056", "me02.5-060", "me02.5-066", "me02.5-070", "me02.5-072",
        "me02.5-084", "me02.5-089", "me02.5-099", "me02.5-105", "me02.5-107", "me02.5-113", "me02.5-117", "me02.5-120",
        "me02.5-145", "me02.5-171", "me03-003", "me03-019", "me03-022", "me03-024", "me03-025", "me03-029", "me03-036",
        "me03-066", "me04-022", "me04-122", "me05-060", "sv01-002", "sv01-019", "sv01-031", "sv01-032", "sv01-035",
        "sv01-047", "sv01-062", "sv01-076", "sv01-079", "sv01-081", "sv01-104", "sv01-106", "sv01-114", "sv02-040",
        "sv02-060", "sv02-064", "sv02-065", "sv02-151", "sv03-009", "sv03-025", "sv03-030", "sv03-041", "sv03-047",
        "sv03-052", "sv03-074", "sv03-078", "sv03-082", "sv03-107", "sv03-109", "sv03-153", "sv03-172", "sv03.5-008",
        "sv03.5-009", "sv03.5-010", "sv03.5-025", "sv03.5-047", "sv03.5-058", "sv03.5-059", "sv03.5-119", "sv03.5-137",
        "sv03.5-144", "sv03.5-145", "sv03.5-146", "sv04-005", "sv04-015", "sv04-021", "sv04-022", "sv04-027",
        "sv04-037", "sv04-038", "sv04-042", "sv04-054", "sv04-058", "sv04-063", "sv04-072", "sv04-093", "sv04-099",
        "sv04-100", "sv04-104", "sv04-108", "sv04.5-012", "sv04.5-026", "sv04.5-042", "sv04.5-043", "sv04.5-067",
        "sv05-016", "sv05-054", "sv05-059", "sv05-068", "sv05-092", "sv05-095", "sv05-118", "sv06-005", "sv06-011",
        "sv06-016", "sv06-022", "sv06-025", "sv06-033", "sv06-095", "sv06-096", "sv06-123", "sv06-124", "sv06.5-005",
        "sv06.5-012", "sv06.5-047", "sv07-014", "sv07-030", "sv07-042", "sv07-045", "sv07-050", "sv07-107", "sv07-118",
        "sv08-021", "sv08-034", "sv08-059", "sv08-065", "sv08-068", "sv08-075", "sv08-079", "sv08-156", "sv08.5-005",
        "sv08.5-011", "sv08.5-012", "sv08.5-044", "sv08.5-045", "sv08.5-049", "sv08.5-056", "sv08.5-070", "sv08.5-085",
        "sv09-007", "sv09-032", "sv09-041", "sv09-047", "sv09-053", "sv09-055", "sv09-073", "sv09-115", "sv10-018",
        "sv10-028", "sv10-036", "sv10-055", "sv10-056", "sv10-060", "sv10-101", "sv10-140", "sv10-145", "sv10.5b-016",
        "sv10.5b-031", "sv10.5b-033", "sv10.5b-050", "sv10.5b-053", "sv10.5b-063", "sv10.5b-067", "sv10.5w-013",
        "sv10.5w-019", "sv10.5w-025", "sv10.5w-034", "sv10.5w-047", "sv10.5w-048"
    ),
    "hp": (
        "me01-002", "me01-005", "me01-006", "me01-007", "me01-008", "me01-010", "me01-011", "me01-013", "me01-014",
        "me01-016", "me01-021", "me01-023", "me01-024", "me01-025", "me01-026", "me01-027", "me01-028", "me01-030",
        "me01-031", "me01-032", "me01-034", "me01-038", "me01-041", "me01-042", "me01-046", "me01-047", "me01-053",
        "me01-064", "me01-095", "me01-097", "me01-103", "me01-105", "me01-106", "me02-001", "me02-025", "me02-032",
        "me02.5-001", "me02.5-010", "me02.5-016", "me02.5-043", "me02.5-058", "me02.5-070", "me02.5-073", "me02.5-076",
        "me02.5-079", "me02.5-089", "me02.5-121", "me02.5-145", "me02.5-148", "me02.5-160", "me03-007", "me03-016",
        "me03-022", "me04-122", "sv01-016", "sv01-019", "sv01-086", "sv01-088", "sv01-091", "sv01-100", "sv02-037",
        "sv02-040", "sv02-043", "sv02-052", "sv02-055", "sv02-061", "sv02-148", "sv02-150", "sv02-166", "sv03-022",
        "sv03-066", "sv03-079", "sv03-082", "sv03-153", "sv03.5-006", "sv03.5-040", "sv03.5-065", "sv03.5-076",
        "sv03.5-115", "sv03.5-124", "sv03.5-130", "sv03.5-145", "sv04-046", "sv04-135", "sv05-034", "sv05-111",
        "sv06-025", "sv06-040", "sv06-061", "sv07-076", "sv07-082", "sv07-105", "sv08-130", "sv08.5-034"
    ),
    "damage": (
        "me01-002", "me01-003", "me01-005", "me01-006", "me01-008", "me01-010", "me01-011", "me01-013", "me01-014",
        "me01-016", "me01-020", "me01-022", "me01-024", "me01-025", "me01-028", "me01-030", "me01-031", "me01-033",
        "me01-034", "me01-038", "me01-040", "me01-041", "me01-044", "me01-047", "me01-051", "me01-052", "me01-053",
        "me01-057", "me01-058", "me01-061", "me01-062", "me01-064", "me01-065", "me01-069", "me01-070", "me01-073",
        "me01-079", "me01-094", "me01-095", "me01-098", "me01-099", "me01-101", "me01-103", "me01-106", "me01-112",
        "me02-004", "me02-007", "me02-010", "me02-014", "me02-015", "me02-016", "me02-022", "me02-025", "me02-030",
        "me02-031", "me02-072", "me02-076", "me02.5-010", "me02.5-013", "me02.5-019", "me02.5-025", "me02.5-028",
        "me02.5-031", "me02.5-040", "me02.5-043", "me02.5-061", "me02.5-064", "me02.5-070", "me02.5-073", "me02.5-076",
        "me02.5-079", "me02.5-082", "me02.5-121", "me02.5-148", "me02.5-152", "me02.5-154", "me02.5-157", "me02.5-160",
        "me02.5-163", "me02.5-166", "me03-025", "me03-028", "me03-031", "me03-034", "me03-040", "me03-041", "me03-043",
        "me03-057", "me03-058", "me03-061", "me04-048", "sv01-010", "sv01-019", "sv01-026", "sv01-034", "sv01-061",
        "sv01-064", "sv01-076", "sv01-085", "sv01-086", "sv01-088", "sv01-106", "sv01-124", "sv01-139", "sv01-142",
        "sv01-145", "sv02-008", "sv02-037", "sv02-061", "sv02-079", "sv03-088", "sv03-153", "sv03-172", "sv03.5-006",
        "sv03.5-010", "sv03.5-073", "sv03.5-076", "sv03.5-112", "sv03.5-115", "sv03.5-149", "sv04-100", "sv04-109",
        "sv04-135", "sv04.5-040", "sv05-034", "sv05-049", "sv06-040", "sv06-061", "sv06.5-006", "sv07-007", "sv07-028",
        "sv07-105", "sv08-048", "sv08-127", "sv08-238", "sv08.5-076", "sv09-079", "sv09-085", "sv09-114", "sv10.5b-034",
        "sv10.5w-020"
    ),
    "weakness": (
        "me01-002", "me01-003", "me01-005", "me01-006", "me01-007", "me01-008", "me01-010", "me01-011", "me01-013",
        "me01-014", "me01-016", "me01-022", "me01-046", "me01-047", "me01-049", "me01-051", "me01-052", "me01-053",
        "me01-095", "me01-097", "me01-098", "me01-099", "me01-103", "me01-105", "me01-106", "me01-107", "me01-108",
        "me01-109", "me01-110", "me01-112", "me02-001", "me02-004", "me02-007", "me02-010", "me02-030", "me02-031",
        "me02-032", "me02-034", "me02-070", "me02-071"
    ),
    "resistance": (
        "me01-103", "me01-105", "me02-070", "me02.5-079", "me02.5-089", "me02.5-145", "me02.5-169", "me02.5-171",
        "me03-055", "me05-065", "sv01-079", "sv01-086", "sv01-088", "sv01-148", "sv02-082", "sv02-163", "sv02-164",
        "sv02-169", "sv03-153", "sv03-163", "sv03-169", "sv03.5-016", "sv03.5-022", "sv03.5-065", "sv03.5-085",
        "sv03.5-142", "sv03.5-145", "sv03.5-151", "sv04-076", "sv04-135", "sv04-139", "sv04.5-022", "sv05-111",
        "sv05-127", "sv05-133", "sv05-139", "sv06-065", "sv06-076", "sv06-094", "sv06-139"
    ),
}


class Glyphs(NamedTuple):
    shapes: dict        # glyph -> float32 coverage at UP x
    x: dict             # glyph -> UP x column of its shape's left edge with nothing set right of it
    y: dict             # glyph -> UP x row of its shape's top (HP: on the card; damage: from the name's baseline)
    advance: dict       # glyph -> UP x px it moves what is set left of it
    kern: dict          # (left glyph, right glyph) -> UP x px it moves the left one further


# ---------------------------------------------------------------- setting
def place(glyphs, seq):
    """[(glyph, UP x column, UP x row)] of a glyph sequence set right-aligned, as fitted."""
    out = []
    for j, ch in enumerate(seq):
        right = seq[j + 1:]
        x = glyphs.x[ch] - sum(glyphs.advance.get(c, 0.0) for c in right) \
            - sum(glyphs.kern.get((a, b), 0.0) for a, b in zip(seq[j:], seq[j + 1:]))
        out.append((ch, x, glyphs.y[ch]))
    return out


def sequence(kind, text):
    return [LABEL, *text] if kind == "hp" else list(text)


def coverage(kind, text, base=0.0):
    """HxW coverage of text ('hp': the figures, with their label; 'damage': figures and suffix; 'weakness' or
    'resistance': the row's value) as the print sets it; base: the attack name's baseline (damage). None when there
    are no glyphs or one is missing."""
    glyphs = load(kind)
    seq = sequence(kind, text)
    if glyphs is None or not text or any(c not in glyphs.shapes for c in seq):
        return None
    return render([(glyphs.shapes[ch], x, y + base * UP) for ch, x, y in place(glyphs, seq)])


def symbol(type_name, cx, base, scale=1.0):
    """HxW coverage of a type's energy symbol in effect text, its gap centred on cx in the line on baseline base;
    scale: the body text's size over the full size. None without lifted symbols."""
    glyphs = load("energy")
    if glyphs is None or type_name not in glyphs.shapes:
        return None
    shape = glyphs.shapes[type_name]
    if scale != 1.0:
        size = (max(1, round(shape.shape[1] * scale)), max(1, round(shape.shape[0] * scale)))
        shape = cv2.resize(shape, size, interpolation=cv2.INTER_AREA)
    return render([(shape, cx * UP + glyphs.x[type_name] * scale, base * UP + glyphs.y[type_name] * scale)])


def render(pieces):
    """HxW coverage of UP x shapes at UP x top-left corners, the most of them where they overlap."""
    out = np.zeros((H, W), np.float32)
    placed = [(shape, round(x), round(y)) for shape, x, y in pieces]
    if not placed:
        return out
    x0 = min(x for _, x, _ in placed) // UP * UP
    y0 = min(y for _, _, y in placed) // UP * UP
    x1 = -(-max(x + s.shape[1] for s, x, _ in placed) // UP) * UP
    y1 = -(-max(y + s.shape[0] for s, _, y in placed) // UP) * UP
    tile = np.zeros((y1 - y0, x1 - x0), np.float32)
    for shape, x, y in placed:
        region = tile[y - y0:y - y0 + shape.shape[0], x - x0:x - x0 + shape.shape[1]]
        np.maximum(region, shape, out=region)
    small = tile.reshape(tile.shape[0] // UP, UP, tile.shape[1] // UP, UP).mean(axis=(1, 3))
    X0, Y0 = x0 // UP, y0 // UP
    sx, sy = max(0, -X0), max(0, -Y0)
    X1, Y1 = min(W, X0 + small.shape[1]), min(H, Y0 + small.shape[0])
    if X1 > X0 + sx and Y1 > Y0 + sy:
        out[Y0 + sy:Y1, X0 + sx:X1] = small[sy:Y1 - Y0, sx:X1 - X0]
    return out


# ---------------------------------------------------------------- lifting
def standout(scan, box, ink_box=None):
    """How far the print stands out from the panel in box, in units of its ink (0..1, HxW), its sign (+1 dark
    print, -1 light) and the spread of the panel round it. The sign is the one standing out the more; the ink is
    read inside what stands out over half as far as the most."""
    x0, y0, x1, y1 = box
    lum = luminance(np.clip(scan[..., :3], 0, 255)).astype(np.float32)
    background = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), BACKGROUND).astype(np.float32)
    best = None
    for sign in (1, -1):
        d = (background - lum)[y0:y1, x0:x1] * sign
        most = float(np.percentile(d, 99.5))
        if best is None or most > best[0]:
            best = (most, sign, d)
    most, sign, d = best
    strokes = d > most / 2
    ink = float(np.percentile(d[strokes], INK_SHARE)) if strokes.any() else 1.0
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = np.clip(d / max(ink, 1.0), 0, 1)
    printed = cv2.dilate((out > 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    clear = ~printed
    clear[:y0], clear[y1:], clear[:, :x0], clear[:, x1:] = False, False, False, False
    noise = float(np.std((d / max(ink, 1.0))[clear[y0:y1, x0:x1]])) if clear.any() else 1.0
    return out, sign, noise


def shapes(cov, min_h=1, max_h=999, min_px=30, max_w=999):
    """[(x, y, w, h, mask)] of the print's shapes, left to right."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((cov > 0.5).astype(np.uint8))
    out = [(*stats[i, :4], labels == i) for i in range(1, n)
           if min_h <= stats[i, 3] <= max_h and stats[i, 4] >= min_px and stats[i, 2] <= max_w]
    return sorted(out, key=lambda s: s[0])


def cut(cov, mask, x, y, w, h):
    """(UP x coverage of one glyph, UP x top-left on the card)."""
    keep = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    x0, y0 = x - PAD, y - PAD
    piece = (cov * keep)[y0:y + h + PAD, x0:x + w + PAD]
    up = cv2.resize(piece, (piece.shape[1] * UP, piece.shape[0] * UP), interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0, 1), (x0 * UP, y0 * UP)


def hp_samples(scan, hp):
    """[(sequence, index, (shape, top-left))] of a card's printed HP: its figures, then its label."""
    cov, sign, noise = standout(scan, HP_BOX)
    if sign < 0 or noise > NOISE:
        return []
    figures = shapes(cov, min_h=HP_FIGURE_H)
    if len(figures) != len(hp):
        return []
    letters = [s for s in shapes(cov, 8, HP_FIGURE_H - 6) if s[0] + s[2] < figures[0][0] and s[1] > 60][-2:]
    seq = (LABEL, *hp)
    out = [(seq, j + 1, cut(cov, s[4], *s[:4])) for j, s in enumerate(figures)]
    if len(letters) == 2:
        x0, y0 = min(s[0] for s in letters), min(s[1] for s in letters)
        x1, y1 = max(s[0] + s[2] for s in letters), max(s[1] + s[3] for s in letters)
        out.append((seq, 0, cut(cov, letters[0][4] | letters[1][4], x0, y0, x1 - x0, y1 - y0)))
    return out


def damage_samples(scan, damages):
    """[(sequence, index, (shape, top-left), name baseline)] of a card's printed damages, top to bottom.
    damages: the damage of each attack that prints one, in order."""
    cov, sign, noise = standout(scan, DAMAGE_BOX)
    if sign < 0 or noise > NOISE:
        return []
    figures = shapes(cov, *DAMAGE_FIGURE_H, max_w=DAMAGE_FIGURE_W)
    lines = []
    for s in figures:
        line = next((l for l in lines if abs((l[-1][1] + l[-1][3]) - (s[1] + s[3])) <= 3), None)
        line.append(s) if line else lines.append([s])
    lines = sorted((l for l in lines if max(s[0] + s[2] for s in l) >= DAMAGE_RIGHT), key=lambda l: l[0][1])
    if len(lines) != len(damages):
        return []
    small = shapes(cov, 2, DAMAGE_FIGURE_H[0] - 1, min_px=12)
    out = []
    for line, damage in zip(lines, damages):
        line = sorted(line, key=lambda s: s[0])
        top, bottom = min(s[1] for s in line), max(s[1] + s[3] for s in line)
        right = max(s[0] + s[2] for s in line)
        suffix = [s for s in small if s[0] >= right - 2 and top - 4 <= s[1] + s[3] / 2 <= bottom + 4]
        figures_of = "".join(c for c in damage if c.isdigit())
        if len(line) != len(figures_of) or len(suffix) != len(damage) - len(figures_of):
            return []
        base = name_baseline(scan, line[0][0], top, bottom)
        if base is None:
            return []
        seq = tuple(damage)
        out += [(seq, j, cut(cov, s[4], *s[:4]), base) for j, s in enumerate(line + suffix)]
    return out


def row_samples(scan, value, box):
    """[(sequence, index, (shape, top-left))] of a weakness-row value (×2, -30) printed in box."""
    cov, sign, noise = standout(scan, box)
    if sign < 0 or noise > NOISE:
        return []
    x0, _, x1, _ = box
    found = [s for s in shapes(cov, 2, 30, min_px=8) if s[0] > x0 and s[0] + s[2] < x1]
    if len(found) != len(value):
        return []
    seq = tuple(value)
    return [(seq, j, cut(cov, s[4], *s[:4])) for j, s in enumerate(found)]


def symbol_samples(card, scan):
    """[(sequence, index, (shape, top-left from its gap's centre and the line's baseline))] of the energy symbols
    printed in a card's effect text (card as modern.read_card gives it), where they are found."""
    from . import compose, modern                  # they set text with these glyphs
    marks = []

    def record(type_name, cx, base, size):
        marks.append((type_name, cx, base, size))
        return np.zeros((H, W), np.float32)
    text = modern._drawn(modern._moves, card, compose.WHITE, None, record).get("text")
    if not marks or any(size != compose.BODY_SIZE for *_, size in marks):
        return []                                  # shrunk text: the print may have shrunk it otherwise
    cov, sign, noise = standout(scan, modern.MOVES_ZONE[card["family"]])
    if sign < 0 or noise > NOISE:
        return []
    middle = compose.ts.cap_height(compose.BODY_FACE, compose.BODY_SIZE) / 2
    out = []
    for type_name, cx, base, _ in marks:
        gap = find_gap(text, cov, cx, base - middle)
        if gap is None:
            continue
        gx, dy = gap
        gy = base + dy
        half, above, below = SYMBOL_BOX
        x0, y0, x1, y1 = gx - half, gy - above, gx + half, gy + below
        inside = [s for s in shapes(cov, min_px=4)
                  if s[0] > x0 and s[1] > y0 and s[0] + s[2] < x1 and s[1] + s[3] < y1]
        if not inside:
            continue
        mask = np.logical_or.reduce([s[4] for s in inside])
        left, top = min(s[0] for s in inside), min(s[1] for s in inside)
        right, bottom = max(s[0] + s[2] for s in inside), max(s[1] + s[3] for s in inside)
        shape, (cx4, cy4) = cut(cov, mask, left, top, right - left, bottom - top)
        out.append(((type_name,), 0, (shape, (cx4 - round(gx * UP), cy4 - round(gy * UP)))))
    return out


def find_gap(text, cov, cx, cy):
    """The print's (gap centre x, offset down) of a gap our text has at (cx, cy): our words either side of it
    matched to the print, both needing to land on the same line. None when they do not match."""
    cx, cy = round(cx), round(cy)
    landed = []
    for xa, xb in ((cx - WORDS, cx - GAP_HALF), (cx + GAP_HALF, cx + WORDS)):
        words = text[cy - LINE_HALF:cy + LINE_HALF, xa:xb].astype(np.float32)
        area = cov[cy - LINE_HALF - SEARCH[1]:cy + LINE_HALF + SEARCH[1], xa - SEARCH[0]:xb + SEARCH[0]]
        if words.std() < 0.05 or area.shape[0] < words.shape[0] or area.shape[1] < words.shape[1]:
            return None
        _, score, _, (lx, ly) = cv2.minMaxLoc(cv2.matchTemplate(area, words, cv2.TM_CCOEFF_NORMED))
        if score < MATCH:
            return None
        landed.append((lx - SEARCH[0], ly - SEARCH[1]))
    (dxl, dyl), (dxr, dyr) = landed
    if abs(dyl - dyr) > 1:
        return None
    return cx + (dxl + dxr) / 2, (dyl + dyr) / 2


def name_baseline(scan, damage_left, top, bottom):
    """Row under the attack name's glyphs (most sit on the baseline), left of a damage on the same line."""
    box = (NAME_LEFT, max(0, top - 12), damage_left - 20, bottom + 8)
    cov, _, _ = standout(scan, box)
    glyphs = shapes(cov, *NAME_GLYPH_H, min_px=20)
    if len(glyphs) < 4:
        return None
    return float(np.median([s[1] + s[3] for s in glyphs]))


def average(samples):
    """{glyph: (shape, [(sequence, index, UP x top-left of the shape's frame)])}: each glyph's samples aligned on
    their running mean and averaged; halfway, samples unlike the mean (a symbol's gap matched to the wrong words)
    are dropped."""
    by_glyph = {}
    for seq, j, (shape, corner) in samples:
        by_glyph.setdefault(seq[j], []).append((seq, j, shape, corner))
    out = {}
    for ch, items in by_glyph.items():
        h = max(s.shape[0] for _, _, s, _ in items) + 16 * UP
        w = max(s.shape[1] for _, _, s, _ in items) + 16 * UP
        mean = np.zeros((h, w), np.float32)
        first = items[0][2]
        mean[8 * UP:8 * UP + first.shape[0], 8 * UP:8 * UP + first.shape[1]] = first
        for round_ in range(ALIGN_ROUNDS):
            total, frames, kept = np.zeros_like(mean), [], []
            for seq, j, shape, (cx, cy) in items:
                _, _, _, (lx, ly) = cv2.minMaxLoc(cv2.matchTemplate(mean, shape, cv2.TM_CCORR))
                under = mean[ly:ly + shape.shape[0], lx:lx + shape.shape[1]]
                agree = float((under * shape).sum() / (np.sqrt((under ** 2).sum() * (shape ** 2).sum()) + 1e-6))
                if round_ == ALIGN_ROUNDS // 2 and agree < AGREE and len(items) > 2:
                    continue
                kept.append((seq, j, shape, (cx, cy)))
                total[ly:ly + shape.shape[0], lx:lx + shape.shape[1]] += shape
                frames.append((seq, j, (cx - lx, cy - ly)))
            items = kept
            mean = total / len(items)
        out[ch] = (mean, frames)
    return out


def fit(averaged):
    """Glyphs: the averaged shapes and where they go. x: least squares over every sample of
    corner = x[glyph] - advances right of it - kerns from it rightwards; y: each glyph's median row."""
    names = sorted(averaged)
    pairs = {}
    for _, frames in averaged.values():
        for seq, j, _ in frames:
            for p in zip(seq[j:], seq[j + 1:]):
                pairs[p] = pairs.get(p, 0) + 1
    kerned = sorted(p for p, n in pairs.items() if n >= MIN_PAIR)
    n_x, n_adv = len(names), len(names)
    rows, values = [], []
    for ch in names:
        for seq, j, (cx, _) in averaged[ch][1]:
            row = np.zeros(n_x + n_adv + len(kerned))
            row[names.index(ch)] = 1
            for c in seq[j + 1:]:
                row[n_x + names.index(c)] -= 1
            for p in zip(seq[j:], seq[j + 1:]):
                if p in kerned:
                    row[n_x + n_adv + kerned.index(p)] -= 1
            rows.append(row)
            values.append(cx)
    ridge = np.zeros((n_adv + len(kerned), len(rows[0])))
    prior = np.zeros(len(ridge))
    for i in range(n_adv):
        ridge[i, n_x + i], prior[i] = ADVANCE_RIDGE, ADVANCE_RIDGE * ADVANCE_PRIOR
    for i in range(len(kerned)):
        ridge[n_adv + i, n_x + n_adv + i] = KERN_RIDGE
    solution, *_ = np.linalg.lstsq(np.vstack([rows, ridge]), np.concatenate([values, prior]), rcond=None)
    return Glyphs(shapes={ch: averaged[ch][0] for ch in names},
                  x=dict(zip(names, solution[:n_x].tolist())),
                  y={ch: float(np.median([cy for _, _, (_, cy) in averaged[ch][1]])) for ch in names},
                  advance=dict(zip(names, solution[n_x:n_x + n_adv].tolist())),
                  kern=dict(zip(kerned, solution[n_x + n_adv:].tolist())))


def lift(kind, card_ids):
    """Glyphs lifted from these cards' scans (TCGdex gives what they print)."""
    samples = []
    for card_id in card_ids:
        card, scan = tcg.card(card_id), tcg.scan(card_id)
        if kind == "hp":
            samples += hp_samples(scan, str(card.get("hp", "")))
        elif kind == "energy":
            from . import modern
            try:
                samples += symbol_samples(modern.read_card(card_id), scan)
            except SystemExit:                     # not a card the editor sets text on
                continue
        elif kind in ROW_BOXES:
            side = (card.get("weaknesses" if kind == "weakness" else "resistances") or [{}])[0]
            value = str(side.get("value") or "").replace("x", "×")
            if value:
                samples += row_samples(scan, value, ROW_BOXES[kind])
        else:
            damages = [str(a.get("damage") or "").replace("x", "×") for a in card.get("attacks") or []]
            for seq, j, piece, base in damage_samples(scan, [d for d in damages if d]):
                shape, (cx, cy) = piece
                samples.append((seq, j, (shape, (cx, cy - round(base * UP)))))
    return fit(average(samples)) if samples else None


# ---------------------------------------------------------------- cache
def _files(kind):
    folder = f"lifted-v{VERSION}"
    return cache.path("assets", folder, f"{kind}.npz"), cache.path("assets", folder, f"{kind}.json")


@functools.lru_cache(maxsize=None)
def load(kind):
    """The glyphs of 'hp', 'damage', 'weakness', 'resistance' or 'energy', lifted once and cached; None if none
    could be."""
    shapes_file, model_file = _files(kind)
    if not model_file.exists():
        glyphs = lift(kind, SOURCES[kind])
        if glyphs is None:
            return None
        names = sorted(glyphs.shapes)
        np.savez_compressed(shapes_file, *[glyphs.shapes[n] for n in names])
        model_file.write_text(json.dumps({
            "names": names, "x": glyphs.x, "y": glyphs.y, "advance": glyphs.advance,
            "kern": [[a, b, v] for (a, b), v in glyphs.kern.items()]}, ensure_ascii=False), encoding="utf-8")
    model = json.loads(model_file.read_text(encoding="utf-8"))
    with np.load(shapes_file) as data:
        stored = [data[f"arr_{i}"] for i in range(len(model["names"]))]
    return Glyphs(shapes=dict(zip(model["names"], stored)), x=model["x"], y=model["y"], advance=model["advance"],
                  kern={(a, b): v for a, b, v in model["kern"]})
