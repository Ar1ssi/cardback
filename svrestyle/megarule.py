"""The Mega Evolution rule ("When 1 of your Pokémon becomes a Mega Evolution Pokémon, your turn ends.", the rule every
XY "M ... EX" card prints in its own box), set the way Gen 9 sets such a rule: the editor's "mega_rule".

Gen 9 prints two kinds of rule. Prize rules (ex, Mega ex) sit in the box at the bottom right; a rule about how the
card plays, Tera's, hangs under the name bar on the art: a pearl panel with an iridescent rim, its label a tab on the
bar above it (sv03-125 Charizard ex, sv06-025 Teal Mask Ogerpon ex, at the same place on Basic and Stage cards). Mega
Evolution cards print their Mega mark at that place too, the "The Mega-Evolved form of ..." badge (me01, me02), so
the turn-ending Mega rule is a Tera-style panel under the "Evolves from" strip with the badge as its label: the badge
sits on the panel's top-right corner, the rule's first line beside it and its second under it, as the bottom rule box
sets its first line beside its pill. The panel is as long as its two lines (the rule is shorter than Tera's), its
rim and body take the badge's colours (its lettering's pink -> violet -> blue -> green, or a Mega Hyper Rare's gold)
along its length, and the portrait ring stays on top of its left end. It covers no abilities or attacks.

A Mega ex print keeps its own badge. Other cards get one rebuilt from SOURCES (badge()): their median, the lettering
taken off the band (its colour along x times its shading down y), "The Mega-Evolved form of", which every print sets
alike, put back, and the Pokemon's name set in white with an outline in the lettering's colours. A card that is not a
Mega gets "Mega Evolution rule" on the badge instead.
"""
import cv2
import numpy as np

from . import cache, layers, meframe, stage, tcg
from . import typeset as ts
from .imaging import fill_holes, grow, hsv, load_rgba, save_rgba

W, H = 733, 1024
VERSION = 1
KIND = "mega_rule"                 # the layers kind of what is drawn (foil plates cut it out)
RULE = "When 1 of your Pokémon becomes a Mega Evolution Pokémon, your turn ends."
# Mega ex Double Rares of me01 / me02: one badge, one place; only the name on its second line differs.
SOURCES = ("me01-003", "me01-022", "me01-036", "me01-050", "me01-060", "me01-077", "me01-086", "me01-094",
           "me01-100", "me01-104", "me02-004", "me02-013", "me02-041", "me02-056", "me02-061", "me02-084")
BADGE = (400, 92, 733, 142)        # x0, y0, x1, y1 round the badge (rows 95-139 on every print)
AGREE_DIFF = 25                    # RGB levels within which a print agrees with the median
BADGE_AGREE = 0.55                 # share of prints agreeing that makes a pixel the badge's (its rims: 0.8-0.9, a
#                                    scan's pixel of play; what the names leave enclosed is filled in)
BADGE_LEFT = 412                   # x: the "Evolves from" strip, which Basic Megas do not print, runs under the badge
#                                    to here (agreeing 0.7-0.8); the Mega symbol's lobe starts at 413
LOBE_END = 462                     # x where the Mega symbol's lobe ends and the lettering begins
LETTER_ROWS = (100, 134)           # rows of the lettering: the white lines along the badge's rims stay
LINE1_ROWS = (100, 118)            # "The Mega-Evolved form of"; the name runs 118-133
SHADE_COLS = (692, 704)            # x: plain band right of the lettering, where the shading down y is read
LETTER_GROW = 2                    # px round a letter (its soft edge) taken off with it
BAND_SMOOTH = 3                    # px the band's colour along x is smoothed over
LETTER_X = (462, 690)              # the lettering's run, whose colours the panel takes along its length
OUTLINE_SAT, OUTLINE_VAL = 0.55, 0.35   # a lettering outline pixel: this saturated, this bright (gold too)
NAME = dict(face=ts.HEAVY_ITALIC, size=17.5, x=463.0, base=131.5, right=700.0, outline=2.1)
LABEL_BASE = 123.0                 # a badge with one line ("Mega Evolution rule"): its baseline
# The panel, fitted on Tera's (sv03-125, sv06-025): top under the strip, left end rounded and clear of the portrait
# ring, right end leaning like the strip's; rule text Gill Sans Bold, Tera's size and line positions.
TOP, BOTTOM, LEFT = 117.5, 166.0, 125.0
CORNER = (16.0, 12.0)              # px: radius of the left end's top and bottom corners
SLANT = 0.85                       # the right end leans this far right per row up
PAD = 14.0                         # px past the second line's end
MIN_REACH = 26.0                   # px the panel's top runs under the badge, past its lobe's lowest left edge
TEXT = dict(face=ts.BOLD, size=15.0, xscale=0.95, bases=(137.5, 157.0), x=(146.0, 137.0))
BADGE_GAP = 12.0                   # px between the first line and the badge
INK = (28, 26, 32)
PEARL = ((252, 251, 253), (233, 232, 238))      # body, top and bottom
SWEEP = 0.10                       # share of the badge's band colours in the body
RIM, RIM_LIGHT, INNER_LIGHT = 2.6, 0.12, 0.7    # px rim in the lettering's colours, lightened; white line inside it
SHADOW = ((30, 26, 48), 1.4, 0.6)  # the dark hairline outside the panel: colour, px, opacity
SS = 4                             # supersampling of the panel's outline


# ---------------------------------------------------------------- the badge
def _asset():
    return cache.path("assets", f"megabadge-v{VERSION}.png")


def _box(img):
    x0, y0, x1, y1 = BADGE
    return img[y0:y1, x0:x1]


def letters_of(rgb):
    """bool over BADGE: the lettering (white fill, its saturated outline), the Mega symbol's lobe left out."""
    _, sat, val = hsv(rgb)
    x0, y0 = BADGE[:2]
    out = np.zeros(rgb.shape[:2], bool)
    rows = slice(LETTER_ROWS[0] - y0, LETTER_ROWS[1] - y0)
    out[rows, LOBE_END - x0:] = ((sat > 0.42) | ((sat < 0.12) & (val > 0.93)))[rows, LOBE_END - x0:]
    return out


def blank_band(rgb, alpha, also=None):
    """rgb over BADGE with its lettering (and `also`, bool) taken off: the band's colour along x times its shading
    down y, each read off what is left clear."""
    x0, y0 = BADGE[:2]
    inside = cv2.erode((alpha > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gone = (cv2.dilate(letters_of(rgb).astype(np.uint8), np.ones((2 * LETTER_GROW + 1,) * 2, np.uint8)) > 0) & inside
    if also is not None:
        gone |= also & inside
    c0, c1 = SHADE_COLS[0] - x0, SHADE_COLS[1] - x0
    rows = inside[:, c0:c1].all(axis=1)
    profile = rgb[:, c0:c1].mean(axis=1)
    shade = np.where(rows[:, None], profile / profile[rows].mean(axis=0), 1.0)
    colour = np.full((rgb.shape[1], 3), np.nan, np.float32)
    clear = inside & ~gone
    for x in range(LOBE_END - x0, rgb.shape[1]):
        if clear[:, x].sum() >= 3:
            colour[x] = np.median(rgb[clear[:, x], x] / shade[clear[:, x]], axis=0)
    known = np.flatnonzero(~np.isnan(colour[:, 0]))
    for c in range(3):
        colour[:, c] = np.interp(np.arange(len(colour)), known, colour[known, c])
    colour = cv2.GaussianBlur(colour[None], (0, 0), sigmaX=BAND_SMOOTH, sigmaY=0)[0]
    return np.where(gone[..., None], shade[:, None, :] * colour[None], rgb), gone


def build_badge():
    """RGBA over BADGE: the badge as every print has it, its second line (the name) taken off."""
    scans = np.stack([_box(tcg.scan(card_id))[..., :3] for card_id in SOURCES])
    med = np.median(scans, axis=0)
    agree = (np.abs(scans - med).max(axis=-1) < AGREE_DIFF).mean(axis=0)
    shape = cv2.GaussianBlur(agree.astype(np.float32), (0, 0), 1.0) > BADGE_AGREE
    shape[:, :BADGE_LEFT - BADGE[0]] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(shape.astype(np.uint8))
    if n < 2:
        raise ValueError("no Mega badge found where the Mega ex prints have it")
    shape = fill_holes(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = cv2.GaussianBlur(shape.astype(np.float32), (3, 3), 0.6)
    x0, y0 = BADGE[:2]
    names = np.zeros(alpha.shape, bool)                    # the second line, the prints' names run together: all of it
    names[LINE1_ROWS[1] - y0:LETTER_ROWS[1] - y0, LOBE_END - x0:] = True
    blank, _ = blank_band(med, alpha, also=names)
    line1 = np.zeros(alpha.shape, bool)
    line1[LINE1_ROWS[0] - y0:LINE1_ROWS[1] - y0] = letters_of(med)[LINE1_ROWS[0] - y0:LINE1_ROWS[1] - y0]
    keep = cv2.GaussianBlur((cv2.dilate(line1.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0).astype(np.float32),
                            (3, 3), 0.7)[..., None]
    rgb = blank * (1 - keep) + med * keep
    return np.dstack([rgb, alpha * 255]).astype(np.float32)


def badge():
    """RGBA float32 over BADGE (built once from SOURCES, cached)."""
    out = _asset()
    if out.exists():
        return load_rgba(out)
    sprite = build_badge()
    save_rgba(sprite, out)
    return sprite


def label_of(card):
    """(lines on the badge, whether the first is the printed "The Mega-Evolved form of"): a Mega names the Pokemon it
    is the Mega-Evolved form of (what a MEGA-stage card evolves from, else its name without "Mega")."""
    name = card.get("name", "")
    if card.get("stage") == stage.MEGA and card.get("evolves_from"):
        return [card["evolves_from"]], True
    if name.startswith("Mega "):
        return [name[len("Mega "):]], True
    return ["Mega Evolution rule"], False


def rebuilt_badge(card):
    """(rgb, alpha) over BADGE for a card that prints no badge."""
    sprite = badge()
    x0, y0 = BADGE[:2]
    lines, printed_first = label_of(card)
    canvas = np.zeros((H, W, 3), np.float32)
    rgb = sprite[..., :3]
    if not printed_first:                                  # the band alone: its first line off too
        rgb, _ = blank_band(rgb, sprite[..., 3] / 255)
    canvas[y0:y0 + rgb.shape[0], x0:x0 + rgb.shape[1]] = rgb
    text = lines[0]
    width = ts.text_width(text, NAME["face"], NAME["size"])
    xscale = min(1.0, (NAME["right"] - NAME["x"]) / width)
    cov = ts.coverage([dict(text=text, face=NAME["face"], size=NAME["size"], x=NAME["x"], xscale=xscale,
                            base=NAME["base"] if printed_first else LABEL_BASE)])
    outline = np.clip(grow(cov, NAME["outline"]), 0, 1)[..., None]
    hue = outline_colours(sprite[..., :3])
    canvas = canvas * (1 - outline) + hue[None] * outline
    canvas = canvas * (1 - cov[..., None]) + 255 * cov[..., None]
    return _box(canvas), sprite[..., 3] / 255


def outline_colours(rgb):
    """(W x 3): the badge lettering's outline colour at each card x (rgb over BADGE), held at the lettering's ends."""
    x0, y0 = BADGE[:2]
    _, sat, val = hsv(rgb)
    ink = (sat > OUTLINE_SAT) & (val > OUTLINE_VAL)
    ink[:LETTER_ROWS[0] - y0] = ink[LETTER_ROWS[1] - y0:] = False
    xs, cols = [], []
    for x in range(LETTER_X[0] - x0, LETTER_X[1] - x0):
        if ink[:, x].sum() >= 2:
            xs.append(x0 + x)
            cols.append(np.median(rgb[ink[:, x], x], axis=0))
    if not xs:
        raise ValueError("no lettering found on the Mega badge")
    cols = cv2.GaussianBlur(np.array(cols, np.float32)[None], (0, 0), sigmaX=6, sigmaY=0)[0]
    return np.stack([np.interp(np.arange(W), xs, cols[:, c]) for c in range(3)], axis=1)


def band_colours(rgb):
    """(W x 3): the badge's colour at each card x (rgb over BADGE): the median down its lettering's rows, which the
    band fills more of than the letters (a gold badge's letters are its band's colour)."""
    x0, y0 = BADGE[:2]
    xs = np.arange(LETTER_X[0], LETTER_X[1])
    cols = np.median(rgb[LETTER_ROWS[0] - y0:LETTER_ROWS[1] - y0, xs - x0], axis=0)
    return np.stack([np.interp(np.arange(W), xs, cols[:, c]) for c in range(3)], axis=1)


def prints_badge(card):
    """Whether the card as printed has the Mega badge: every Mega ex print does (all rarities, promos)."""
    return card.get("family") == "ex" and card.get("name", "").startswith("Mega ") \
        and str(card.get("id", "")).startswith("me")


# ---------------------------------------------------------------- the panel
def _arc(cx, cy, r, a0, a1, n=10):
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a1, n)]


def outline(right_bottom):
    """The panel's outline (x, y; pixel r spans r - 0.5 to r + 0.5): rounded left end, right end leaning like '/'
    from right_bottom at the bottom."""
    top_left, bottom_left = CORNER
    return (_arc(LEFT + top_left, TOP + top_left, top_left, np.pi, 1.5 * np.pi)
            + [(right_bottom + SLANT * (BOTTOM - TOP), TOP), (right_bottom, BOTTOM)]
            + _arc(LEFT + bottom_left, BOTTOM - bottom_left, bottom_left, 0.5 * np.pi, np.pi))


def _inside(xs, ys, polygon):
    """bool grid: which points (x across, y down) lie inside the polygon (even-odd rule)."""
    out = np.zeros((len(ys), len(xs)), bool)
    x, y = xs[None, :], ys[:, None]
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if ay != by:
            out ^= ((ay > y) != (by > y)) & (x < ax + (y - ay) * (bx - ax) / (by - ay))
    return out


def coverage(polygon):
    """HxW float32: the share of each pixel inside the polygon."""
    sub = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (np.arange(W)[:, None] + sub).ravel()
    ys = (np.arange(H)[:, None] + sub).ravel()
    return _inside(xs, ys, polygon).reshape(H, SS, W, SS).mean(axis=(1, 3)).astype(np.float32)


def badge_left(alpha):
    """Per badge row: x of its left edge."""
    return BADGE[0] + np.argmax(alpha > 0.5, axis=1)


def split(first_width):
    """The rule in two lines, the first no wider than first_width."""
    words, first = RULE.split(), []
    while words and ts.text_width(" ".join(first + [words[0]]), TEXT["face"], TEXT["size"],
                                  TEXT["xscale"]) <= first_width:
        first.append(words.pop(0))
    return " ".join(first), " ".join(words)


def ring_cover(card_stage):
    """HxW: the portrait ring and its disc, which lie over the panel's left end on a Stage card."""
    if not card_stage:
        return np.zeros((H, W), np.float32)
    cx, cy, rx, ry = meframe.PORTRAIT_RING
    yy, xx = np.mgrid[:H, :W]
    ring = ((xx - cx) / (rx + 1)) ** 2 + ((yy - cy) / (ry + 1)) ** 2 < 1
    return stage.sprite_of(card_stage)[..., 3] / 255 * ring


def along(colours, polygon):
    """(W x 3): colours (along the badge's lettering, LETTER_X) laid along the panel's own length."""
    xs = [x for x, _ in polygon]
    t = np.clip((np.arange(W) - min(xs)) / (max(xs) - min(xs)), 0, 1)
    src = LETTER_X[0] + t * (LETTER_X[1] - LETTER_X[0])
    return np.stack([np.interp(src, np.arange(W), colours[:, c]) for c in range(3)], axis=1)


def draw(canvas, card, printed_badge=False):
    """Set the panel and its badge on canvas (RGBA float32 733x1024). printed_badge: the canvas prints its own badge
    (a Mega ex print), kept on top and read for the colours; else a rebuilt one is set."""
    x0, y0, x1, y1 = BADGE
    if printed_badge:
        alpha = badge()[..., 3] / 255
        rgb = _box(canvas)[..., :3].copy()
    else:
        rgb, alpha = rebuilt_badge(card)
    edge, rows = badge_left(alpha), np.flatnonzero((alpha > 0.5).any(axis=1))
    beside = edge[int(TEXT["bases"][0] - 11) - y0:int(TEXT["bases"][0] + 1) - y0]       # the first line's rows
    lines = split(float(beside.max()) - BADGE_GAP - TEXT["x"][0])
    second = ts.text_width(lines[1], TEXT["face"], TEXT["size"], TEXT["xscale"])
    reach = float(edge[rows[-8:]].min()) + MIN_REACH - SLANT * (BOTTOM - TOP)
    polygon = outline(max(TEXT["x"][1] + second + PAD, reach))
    under_ring = 1 - ring_cover(card.get("stage", 0))
    shape = coverage(polygon) * under_ring
    hue = along(outline_colours(rgb), polygon)
    band = along(band_colours(rgb), polygon)
    t = np.clip((np.arange(H)[:, None, None] - TOP) / (BOTTOM - TOP), 0, 1)
    top, bottom = (np.array(c, np.float32) for c in PEARL)
    body = (top * (1 - t) + bottom * t) * (1 - SWEEP) + band[None] * SWEEP
    solid = cv2.copyMakeBorder((shape > 0.5).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    depth = cv2.distanceTransform(solid, cv2.DIST_L2, 3)[1:-1, 1:-1]
    rim = np.clip(RIM + 0.5 - depth, 0, 1)[..., None]
    light = (np.clip(1 - np.abs(depth - RIM - 0.8), 0, 1) * INNER_LIGHT)[..., None]
    body = body * (1 - rim) + (hue * (1 - RIM_LIGHT) + 255 * RIM_LIGHT)[None] * rim
    body = body * (1 - light) + 255 * light
    colour, px, opacity = SHADOW
    shadow = (np.clip(grow(shape, px) - shape, 0, 1) * opacity * under_ring)[..., None]
    text = ts.coverage([dict(text=line, face=TEXT["face"], size=TEXT["size"], xscale=TEXT["xscale"], x=x, base=base)
                        for line, x, base in zip(lines, TEXT["x"], TEXT["bases"])])
    text = text[..., None]
    c = canvas[..., :3]
    c[:] = c * (1 - shadow) + np.array(colour, np.float32) * shadow
    c[:] = c * (1 - shape[..., None]) + body * shape[..., None]
    c[:] = c * (1 - text) + np.array(INK, np.float32) * text
    a = alpha[..., None]
    c[y0:y1, x0:x1] = c[y0:y1, x0:x1] * (1 - a) + rgb * a
    drawn = np.maximum(np.maximum(shape, shadow[..., 0]), text[..., 0])
    covered = drawn[y0:y1, x0:x1]                # a printed badge keeps its own foil; a rebuilt one is cut out
    drawn[y0:y1, x0:x1] = covered * (1 - alpha) if printed_badge else np.maximum(covered, alpha)
    with layers.kind(KIND):
        layers.record(drawn)
    return lines
