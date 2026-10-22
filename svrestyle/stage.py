"""The Stage pieces top left, as Scarlet & Violet and Mega Evolution print them: the STAGE tab, the silver ring with
the evolution portrait, the "Evolves from" strip, and on regular cards a silver swoosh in the name bar.

The pieces are meframe.stage_sprite's (lifted from me1 Kadabra and Alakazam). They sit at the same place on SV and
ME cards, regular and ex: checked on Kadabra and Alakazam (sv3.5, me1), Alakazam ex and Mega Venusaur ex. Every one
of those prints the same picture of the pre-evolution on the same light disc, so a portrait is lifted whole from a
card TCGdex lists as evolving from that Pokemon. A card id can be named instead: an SV / ME Stage card gives its
portrait, any other card the middle of its art window.

Besides Basic, Stage 1 and Stage 2 there is a custom stage, Mega (MEGA): the Stage 1 pieces with the tab reading MEGA.
Its portrait comes first from the XY "M ... EX" cards, which print the Pokemon they evolve from on a gold disc: the
figure is keyed off the gold and set on the modern silver disc. They are also the fallback for other stages when no
SV / ME card evolves from the Pokemon; after them the Pokemon's official artwork (PokeAPI).
"""
import functools
import os
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import cache, frame, layers, meframe, tcg
from . import typeset as ts
from .imaging import dilate, fill_holes, hsv, load_rgba, save_rgba

W, H = 733, 1024
PORTRAIT_ERAS = ("sv", "me")      # card ids whose Stage cards print the portrait disc
ART_ERAS = PORTRAIT_ERAS + ("swsh", "sm")    # card ids whose art window is meframe.ART
MAX_TRIES = 3                     # candidate cards tried for a portrait before giving up
XY_ERAS = ("xy", "g1")            # XY and Generations: their "M ... EX" cards (TCGdex stage MEGA)
XY_DISC = (80.7, 112.9, 47.4)     # the gold portrait disc on those cards: centre x, y, radius (same on all measured)
XY_GOLD = (30, 62, 0.35, 0.5)     # the disc's gold: hue range (degrees), least saturation and value
XY_FLOOD = 14                     # per-channel step the gold background is flood-filled across (it shades radially)
XY_RIM = 4                        # px inside the disc's edge kept off the figure (the disc's own rim line)
ARTWORK_FILL = 0.8                # an official artwork's longer side, as a share of the disc's width
DISC_RIM, DISC_SMOOTH = 2.5, 6.0  # px: the disc's own rim, kept; blur the light disc is evened out with

# Parts of the stage sprite, by box (x0, y0, x1, y1); the ring is what is left. The strip's right end curls up
# into the regular cards' swoosh; ex cards keep their own bar swoosh. The tab is only its lettering (BASIC, STAGE1,
# STAGE2 sit on the same silver tab), so a card keeps its own tab and border (darker on ex cards).
TAB = meframe.STAGE_TAB
SWOOSH = (470, 0, 600, 94)
STRIP = (128, 86, 520, 126)
SPECK = 40                        # px: smaller bits of a sprite are strays of its cut-out (one left of the ring)
REGULAR = ("tab", "ring", "strip", "disc", "swoosh")
EX = ("tab", "ring", "strip", "disc")
LETTER_BOX = (20, 32, 122, 58)    # x0, y0, x1, y1: the tab's lettering, inside its silver lozenge
TAB_LETTERS = 0.5                 # HSV value under which a lozenge pixel is lettering (dark grey on light silver)
LETTER_GROW = 2                   # px around the letters: their light rim

MEGA = 3                          # the custom Mega stage
# No print has a MEGA tab. Verdana Bold, widened and slanted forward, matches the STAGE lettering best of 25 bold
# faces (Tahoma Bold next; squared G, pointed A): fitted letter by letter on the S, T, A, G and E of the me1 STAGE 1
# and STAGE 2 tabs. Set like them: dark fill, white rim, thin dark edge, the word centred where all three tabs
# centre theirs. Without Verdana, Tahoma Bold, then Gill Sans Bold.
MEGA_FONTS = ("verdanab.ttf", "tahomabd.ttf")
MEGA_CAP, MEGA_XSCALE, MEGA_SHEAR = 10.625, 1.4, 0.18   # px, width factor, slant (x per y, tops to the right)
LETTER_SHIFT = 3                  # px a scan's tab lettering may sit off the me1 tabs'
LETTER_COVER = LETTER_GROW + 1    # px wiped round the letters' fill: their rim, edge and a scan's blur
TAB_SILVER = (0.16, 0.25)         # saturation under, HSV value over: a tab pixel, not art showing by it
MIN_CLEAN = 6                     # tab px of a row free of letters that give the row's shade
MIN_CLEAN_COLUMN = 3              # same for a column
SHADE_SMOOTH = 5                  # px the column shading is averaged over
STRAY = 25                        # RGB levels off the rebuilt tab at which a px by the letters is left-over rim
TAB_SHOWN = 0.6                   # share of the lozenge in tab silver for a canvas to have a tab of its own
MEGA_CENTRE, MEGA_BOTTOM = 69, 50                        # ink box's middle x and last row, as BASIC and STAGE
MEGA_RIM, MEGA_EDGE = 2.2, 0.8                           # px: white rim, then the thin dark edge round it
MEGA_SOFT = 0.3                                          # px blur: the scans' lettering is a touch soft
MEGA_GAP = 2                                             # px between letters at their closest (STAGE / BASIC: 2-3)
MEGA_INK = {"fill": (70, 70, 74), "rim": (255, 252, 255), "edge": (110, 110, 114)}  # sampled on me1 tabs and scans
MEGA_VERSION = 6                                         # bump when the MEGA tab sprite changes (cached)
WORD_SS = 8                                              # supersampling for the tab lettering

# The "Evolves from" line is not Gill Sans but a Frutiger-style italic ('f' without a descender): Segoe UI Bold
# Italic matches it glyph for glyph once narrowed and tightened (fitted on Kadabra and Alakazam of sv3.5 and me1,
# Alakazam ex, Mega Venusaur ex, Mega Clefable ex). Without Segoe UI, Gill Sans Bold Italic as first measured.
EVOLVES_FONT = "segoeuiz.ttf"
EVOLVES_FIT = dict(size=19.6, xscale=0.9, tracking=-0.6, base=112)
EVOLVES_FALLBACK = dict(face=ts.BOLD_ITALIC, size=17.5, xscale=1.0, tracking=0.0, base=111)
EVOLVES_X, EVOLVES_RIGHT = 143, 490
STRIP_PAD = 8                     # px the "Evolves from" line keeps from the strip's end
# Mega Evolution ex cards (me1 Mega Venusaur, me3 Mega Clefable) end the strip under the "Mega-Evolved form of"
# badge, which starts here.
MEGA_STRIP_END = 408


# ---------------------------------------------------------------- pieces
@functools.lru_cache(maxsize=1)
def _boxes():
    def box(b):
        m = np.zeros((H, W), bool)
        x0, y0, x1, y1 = b
        m[y0:y1, x0:x1] = True
        return m
    cx, cy, r = meframe.PORTRAIT_DISC
    yy, xx = np.mgrid[:H, :W]
    disc = np.hypot(xx - cx, yy - cy) < r
    swoosh = box(SWOOSH)
    strip = box(STRIP) & ~swoosh & ~disc
    tab = dilate(letters(basic_tab()) | letters(meframe.stage_sprite(1)) | letters(meframe.stage_sprite(2))
                 | letters(mega_tab()), LETTER_GROW)
    return {"tab": tab, "swoosh": swoosh, "strip": strip, "disc": disc, "ring": ~(box(TAB) | swoosh | strip | disc)}


def under_tab(sprite):
    """bool HxW: what a stage sprite has in the tab box besides the BASIC tab itself (the ring's top)."""
    out = np.zeros((H, W), bool)
    x0, y0, x1, y1 = TAB
    out[y0:y1, x0:x1] = ((sprite[..., 3] > 25) & ~(basic_tab()[..., 3] > 128))[y0:y1, x0:x1]
    return out


def letters(sprite):
    """bool HxW: the tab lettering of a stage sprite or the BASIC tab."""
    _, _, val = hsv(sprite[..., :3])
    out = np.zeros((H, W), bool)
    x0, y0, x1, y1 = LETTER_BOX
    out[y0:y1, x0:x1] = ((val < TAB_LETTERS) & (sprite[..., 3] > 128))[y0:y1, x0:x1]
    return out


def part_mask(parts):
    """bool HxW: where the named parts of a stage sprite lie."""
    out = np.zeros((H, W), bool)
    for p in parts:
        out |= _boxes()[p]
    return out


def basic_tab():
    """RGBA 733x1024: the BASIC tab of me1 Abra (type-independent, like the STAGE tabs)."""
    out = cache.path("assets", f"me-v{meframe.ASSET_VERSION}", "basic-tab.png")
    if out.exists():
        return load_rgba(out)
    card = tcg.scan(meframe.BASIC_PSYCHIC)
    _, sat, val = hsv(card[..., :3])
    alpha = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = TAB
    alpha[y0:y1, x0:x1] = (((sat < 0.16) & (val > 0.25)) | (val < TAB_LETTERS))[y0:y1, x0:x1]
    sprite = card.copy()
    sprite[..., 3] = cv2.GaussianBlur(alpha, (3, 3), 0.6) * 255
    save_rgba(sprite, out)
    return sprite


def sprite_of(stage):
    """RGBA 733x1024: the Stage pieces of stage 1, 2 or MEGA."""
    return mega_tab() if stage == MEGA else meframe.stage_sprite(stage)


def _font_file(name):
    """Path of a font file in the fonts folder or the system font folders, or None."""
    dirs = [ts._font_dir, Path(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
            Path(os.environ.get("LOCALAPPDATA", "~"), "Microsoft", "Windows", "Fonts"),
            "/usr/share/fonts/truetype/msttcorefonts", "/Library/Fonts"]
    return next((str(Path(d, name)) for d in dirs if d and Path(d, name).exists()), None)


def _glyph(font, ch, xscale, shear, strokes):
    """One letter's coverage (h x w, 1x) grown by each stroke width, slanted and widened."""
    w, h, base = int(font.getlength(ch)) + 30 * WORD_SS, 40 * WORD_SS, 30 * WORD_SS
    out = []
    for stroke in strokes:
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((8 * WORD_SS, base), ch, font=font, fill=255, anchor="ls",
                                 stroke_width=round(stroke * WORD_SS), stroke_fill=255)
        img = img.transform(img.size, Image.AFFINE, (1, shear, -shear * base, 0, 1, 0), Image.BICUBIC)
        out.append(np.asarray(img.resize((max(1, round(w * xscale / WORD_SS)), h // WORD_SS), Image.BOX),
                              np.float32) / 255)
    return out


def _word(path, text, cap, xscale, shear, strokes, gap):
    """Coverage (h x w, 1x) of text grown by each stroke width, and the fill's ink left, right and bottom. Letters are
    set one by one, each `gap` px from the last at their closest row, as the printed tabs space them (a font's own
    spacing is wider at this weight and width)."""
    size = 4.0
    while True:                          # font size whose E is `cap` px tall
        box = ImageFont.truetype(path, round(size * WORD_SS)).getbbox("E")
        if (box[3] - box[1]) / WORD_SS >= cap or size > 60:
            break
        size += 0.25
    font = ImageFont.truetype(path, round(size * WORD_SS))
    glyphs = [_glyph(font, ch, xscale, shear, strokes) for ch in text]
    h = glyphs[0][0].shape[0]
    out = [np.zeros((h, sum(g[0].shape[1] for g in glyphs)), np.float32) for _ in strokes]
    right = np.full(h, -1)               # per row, the last column the word's fill reaches so far
    for g in glyphs:
        ink = g[0] > 0.5
        left = np.where(ink.any(axis=1), ink.argmax(axis=1), -1)
        rows = (left >= 0) & (right >= 0)
        if rows.any():
            x = int(np.max(right[rows] - left[rows])) + 1 + gap
        else:                            # first letter, or no row shared with the last: by ink boxes
            x = int(right.max() - left[left >= 0].min()) + 1 + gap if (right >= 0).any() else 0
        for layer, src in zip(out, g):
            region = layer[:, x:x + src.shape[1]]
            np.maximum(region, src[:, :region.shape[1]], out=region)
        cols = np.where(ink.any(axis=1), ink.shape[1] - 1 - ink[:, ::-1].argmax(axis=1), -1)
        right = np.where(cols >= 0, np.maximum(right, x + cols), right)
    ys, xs = np.nonzero(out[0] > 0.5)
    return out, (xs.min(), xs.max(), ys.max())


@functools.lru_cache(maxsize=1)
def _lozenge():
    """bool HxW: the tab's silver lozenge in the letter box (the BASIC tab's), a px short of its outline."""
    inside = fill_holes(basic_tab()[..., 3] > 128) & _letter_box()
    return cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _lettering(image):
    """bool HxW: the tab lettering image prints, with its rim and edge: BASIC, STAGE 1 or STAGE 2 as the me1 tabs
    set them, up to LETTER_SHIFT px off, whichever has its fill darkest against its rim there. A sprite's see-through
    px in the tab count too (Kadabra's cut-out 1)."""
    _, _, val = hsv(image[..., :3])
    inner = _lozenge()
    best, found = -np.inf, None
    for sprite in (basic_tab(), meframe.stage_sprite(1), meframe.stage_sprite(2)):
        fill = letters(sprite)
        for dy in range(-LETTER_SHIFT, LETTER_SHIFT + 1):
            for dx in range(-LETTER_SHIFT, LETTER_SHIFT + 1):
                at = np.roll(fill, (dy, dx), (0, 1)) & inner
                rim = dilate(at, LETTER_GROW) & ~at & inner
                if not at.any() or not rim.any():
                    continue
                contrast = val[rim].mean() - val[at].mean()
                if contrast > best:
                    best, found = contrast, at
    if image.shape[2] == 4:
        found = found | (inner & (image[..., 3] < 128))
    return dilate(found, LETTER_COVER) & inner


def _fill_gaps(shade, known):
    """shade (n x 3) with its unknown entries taken along from the known ones."""
    if not known.any():
        return np.nan_to_num(shade)
    at = np.arange(len(shade))
    for c in range(3):
        shade[~known, c] = np.interp(at[~known], at[known], shade[known, c])
    return shade


def blank_tab(image):
    """RGB 733x1024: image with its tab lettering taken off. The silver shades by row (light top, darker band
    below) plus by column (towards its rounded ends); both are read off the tab pixels clear of the letters, and
    the letters take their sum, as do left-over specks of rim beside them."""
    lettered = _lettering(image)
    _, sat, val = hsv(image[..., :3])
    x0, y0, x1, y1 = LETTER_BOX
    tab = image[y0:y1, x0:x1, :3].astype(np.float32)
    clean = (_lozenge() & ~lettered & (sat < TAB_SILVER[0]) & (val > TAB_SILVER[1]))[y0:y1, x0:x1]
    seen = np.where(clean[..., None], tab, np.nan)
    by_column = np.zeros((x1 - x0, 3), np.float32)
    with warnings.catch_warnings():                       # rows / columns with no clean px: filled after
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(5):
            by_row = _fill_gaps(np.nanmedian(seen - by_column[None], axis=1), clean.sum(1) >= MIN_CLEAN)
            by_column = _fill_gaps(np.nanmedian(seen - by_row[:, None], axis=0), clean.sum(0) >= MIN_CLEAN_COLUMN)
            by_column = cv2.blur(by_column[None], (SHADE_SMOOTH, 1))[0]
    shade = by_row[:, None] + by_column[None]
    stray = (dilate(lettered, LETTER_GROW) & _lozenge())[y0:y1, x0:x1] \
        & (np.abs(tab - shade).max(axis=2) > STRAY)
    wipe = lettered[y0:y1, x0:x1] | stray
    out = image[..., :3].copy()
    out[y0:y1, x0:x1][wipe] = shade[wipe]
    return out


def _letter_box():
    box = np.zeros((H, W), bool)
    x0, y0, x1, y1 = LETTER_BOX
    box[y0:y1, x0:x1] = True
    return box


def _mega_font():
    """Path of the tab lettering's face: the first of MEGA_FONTS found, else Gill Sans Bold."""
    path = next(filter(None, map(_font_file, MEGA_FONTS)), None)
    if path is None:
        warnings.warn(f"{' / '.join(MEGA_FONTS)} not found: the MEGA tab is set in Gill Sans Bold")
        path = f"{ts._font_dir}/{ts.BOLD}.otf"
    return path


def _mega_word():
    return _tab_word("MEGA")


@functools.lru_cache(maxsize=4)
def _tab_word(text):
    """[(coverage HxW, RGB)]: text lettered as the tabs letter theirs, its edge, rim and fill in the order they are
    laid."""
    (fill, rim, edge), (left, right, bottom) = _word(
        _mega_font(), text, MEGA_CAP, MEGA_XSCALE, MEGA_SHEAR,
        (0, MEGA_RIM, MEGA_RIM + MEGA_EDGE), MEGA_GAP)
    dx, dy = MEGA_CENTRE - round((left + right) / 2), MEGA_BOTTOM - bottom
    out = []
    for layer, ink in ((edge, "edge"), (rim, "rim"), (fill, "fill")):
        a = np.zeros((H, W), np.float32)
        lh, lw = layer.shape
        a[dy:dy + lh, dx:dx + lw] = layer[:H - dy, :W - dx]
        out.append((cv2.GaussianBlur(a, (0, 0), MEGA_SOFT) * _letter_box(), np.array(MEGA_INK[ink], np.float32)))
    return out


def mega_letters(image):
    """RGB 733x1024: image's tab with its lettering taken off and MEGA set on it."""
    out = blank_tab(image)
    for a, ink in _mega_word():
        out = out * (1 - a[..., None]) + ink * a[..., None]
    return out


def mega_tab():
    """RGBA 733x1024: the Stage 1 pieces with the tab reading MEGA (cached)."""
    out = cache.path("assets", f"me-v{meframe.ASSET_VERSION}", f"mega-tab-v{MEGA_VERSION}.png")
    if out.exists():
        return load_rgba(out)
    sprite = meframe.stage_sprite(1).copy()
    # the tab is opaque all over its lozenge (Kadabra's cut-out lets parts of the 1 through); holes are filled
    # inside the letter box only, where the notch between the tab and the ring is open to the box's edge
    lozenge = fill_holes((sprite[..., 3] > 128) & _letter_box()) & _letter_box()
    sprite[lozenge, 3] = 255
    sprite[..., :3] = np.where(lozenge[..., None], mega_letters(sprite), sprite[..., :3])
    save_rgba(sprite, out)
    return sprite


def shows_tab(canvas):
    """Whether canvas has a silver BASIC / STAGE tab of its own (a scan or a template), for set_mega."""
    _, sat, val = hsv(canvas[..., :3])
    return float(((sat < TAB_SILVER[0]) & (val > TAB_SILVER[1]))[_lozenge()].mean()) > TAB_SHOWN


def set_mega(canvas):
    """The canvas's own tab made MEGA: it keeps its shading (a scan's print, softness and all), its own letters
    (wherever this scan has them) are taken off and MEGA set. The MEGA sprite's tab is only for a canvas without
    one; Kadabra's cut-out 1 leaves that sprite's right end rebuilt. Recorded as paste records a pasted tab."""
    old = _lettering(canvas)
    canvas[..., :3] = mega_letters(canvas)
    new = _mega_word()[0][0]
    with layers.kind("stage_tab"):
        layers.record(np.maximum(old.astype(np.float32), new))
    with layers.kind("stage_text"):
        layers.record(new)


def paste(canvas, sprite, where):
    """Paste sprite (RGBA) where `where` (bool HxW) allows; recorded as 'stage', what it pastes on the tab as
    'stage_tab', the letters there as 'stage_text'."""
    a = sprite[..., 3] / 255 * where
    n, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0.1).astype(np.uint8))
    a[np.isin(labels, np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < SPECK)) & (labels > 0)] = 0
    canvas[..., :3] = canvas[..., :3] * (1 - a[..., None]) + sprite[..., :3] * a[..., None]
    tab = a * _boxes()["tab"]
    with layers.kind("stage"):
        layers.record((a - tab).astype(np.float32))
    with layers.kind("stage_tab"):
        layers.record(tab.astype(np.float32))
    with layers.kind("stage_text"):
        layers.record((a * letters(sprite)).astype(np.float32))


# ---------------------------------------------------------------- portrait
def _disc_alpha(size, r):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return np.clip(r - 0.5 - np.hypot(xx - c, yy - c), 0, 1).astype(np.float32)


def disc_sprite(card_id):
    """(rgb, alpha) of an SV / ME Stage card's portrait disc."""
    cx, cy, r = meframe.PORTRAIT_DISC
    x0, y0, size = round(cx - r), round(cy - r), round(2 * r)
    crop = tcg.scan(card_id)[y0:y0 + size, x0:x0 + size, :3]
    return crop, _disc_alpha(size, r)


def art_sprite(card_id):
    """(rgb, alpha): the middle of a card's art window as a disc (the Pokemon is usually there)."""
    x0, y0, x1, y1 = meframe.ART
    side = y1 - y0
    left = (x0 + x1 - side) // 2
    crop = tcg.scan(card_id)[y0:y1, left:left + side, :3]
    r = meframe.PORTRAIT_DISC[2]
    size = round(2 * r)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), _disc_alpha(size, r)


def _search(field, name):
    return tcg.search(field, name)


def _official(card_id):
    set_id = card_id.rsplit("-", 1)[0]
    return (tcg.set_info(set_id).get("cardCount") or {}).get("official") or 0


def rank(briefs, eras):
    """Card ids of briefs from these eras, best portrait source first: main-set cards before secret rares (other
    art round the disc), regular before ex, Mega Evolution before Scarlet & Violet, newest set first."""
    def key(b):
        local = re.sub(r"\D", "", b["localId"])
        secret = not local or int(local) > _official(b["id"])
        return (secret, b["name"].endswith(" ex"), not b["id"].startswith("me"), _reverse(b["id"]))
    return [b["id"] for b in sorted((b for b in briefs if b.get("id", "").startswith(eras)), key=key)]


def _reverse(card_id):
    """Sort key that puts later sets first: sv10 before sv09 before sv03.5."""
    m = re.match(r"[a-z]+0*(\d+)(\.5)?", card_id)
    return -(int(m.group(1)) + (0.5 if m.group(2) else 0)) if m else 0


def xy_sprite(card_id):
    """(rgb, alpha) of an XY "M ... EX" card's portrait: the figure on its gold disc, the gold keyed off (flood-filled
    from the disc's edge: the figures are outlined, so the fill stops at them)."""
    cx, cy, r = XY_DISC
    x0, y0, size = round(cx - r), round(cy - r), round(2 * r)
    rgb = tcg.scan(card_id)[y0:y0 + size, x0:x0 + size, :3]
    hue, sat, val = hsv(rgb)
    lo, hi, min_sat, min_val = XY_GOLD
    gold = (hue > lo) & (hue < hi) & (sat > min_sat) & (val > min_val)
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    u8 = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    flood = np.zeros((size + 2, size + 2), np.uint8)
    step = (XY_FLOOD,) * 3
    for y, x in zip(*np.nonzero(gold & (d > r - XY_RIM - 3) & (d < r - XY_RIM))):
        if not flood[y + 1, x + 1]:
            cv2.floodFill(u8, flood, (int(x), int(y)), 0, step, step, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    background = flood[1:-1, 1:-1] > 0
    background |= dilate(background, 2) & gold          # the gold fringe the anti-aliasing leaves round the figure
    figure = (d < r - XY_RIM) & ~background
    if figure.sum() < 0.05 * size * size:
        raise ValueError(f"{card_id}: no portrait figure found on the gold disc")
    return rgb, cv2.GaussianBlur(figure.astype(np.float32), (3, 3), 0.7)


def _base_name(name):
    """Latias-EX, M Latias EX, Latias ex -> latias."""
    name = re.sub(r"^M\s+", "", name.strip())
    return re.sub(r"[\s-]+(EX|ex|GX|V)$", "", name).strip().casefold()


def find_xy_mega(name):
    """(rgb, alpha) from the XY "M ... EX" card of the Pokemon called name, or None."""
    base = _base_name(name)
    briefs = [b for b in _search("stage", "MEGA") if _base_name(b.get("name", "")) == base]
    return _first(rank(briefs, XY_ERAS), xy_sprite)[1]


def pokeapi_sprite(name):
    """(rgb, alpha) of the Pokemon's official artwork (PokeAPI), fitted inside the disc, or None if PokeAPI has no
    such Pokemon."""
    from . import mecompose                      # mecompose imports this module
    words = name.strip().split(" ", 1)
    region = mecompose.REGIONAL.get(words[0]) if len(words) == 2 else None
    species = _base_name(words[1] if region else name)
    slug = re.sub(r"[^a-z0-9-]+", "", species.replace("♀", "-f").replace("♂", "-m").replace(" ", "-"))
    slug = f"{slug}-{region}" if region else slug
    try:
        mon = cache.fetch_json(f"{mecompose.POKEAPI}/pokemon/{slug}", cache.path("pokeapi", f"pokemon-{slug}.json"))
        url = ((mon.get("sprites") or {}).get("other") or {}).get("official-artwork", {}).get("front_default")
        if not url:
            return None
        art = np.asarray(Image.open(cache.fetch(url, cache.path("pokeapi", f"artwork-{slug}.png"))).convert("RGBA"),
                         np.float32)
    except cache.DownloadError:
        return None
    ys, xs = np.nonzero(art[..., 3] > 8)
    art = art[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = 2 * meframe.PORTRAIT_DISC[2] * ARTWORK_FILL
    scale = side / max(art.shape[:2])
    size = (max(1, round(art.shape[1] * scale)), max(1, round(art.shape[0] * scale)))
    art = cv2.resize(art, size, interpolation=cv2.INTER_AREA)
    return art[..., :3], art[..., 3] / 255


def from_card(card_id):
    """(rgb, alpha) from a card the user named: its portrait if it is an SV / ME Stage card or an XY Mega, else its
    art."""
    c = tcg.card(card_id)
    if card_id.startswith(PORTRAIT_ERAS) and c.get("stage") in ("Stage1", "Stage2"):
        return disc_sprite(card_id)
    if card_id.startswith(XY_ERAS) and c.get("stage") == "MEGA":
        return xy_sprite(card_id)
    return art_sprite(card_id)


def _first(ids, sprite):
    """(card id, sprite(card id)) of the first card id that gives one, or (None, None)."""
    for card_id in ids[:MAX_TRIES]:
        try:
            return card_id, sprite(card_id)
        except (cache.DownloadError, ValueError):
            continue
    return None, None


def find(name, mega=False):
    """(rgb, alpha) of the Pokemon called name, or None (with a warning): the SV / ME portrait of a card evolving from
    it, else its XY Mega portrait (tried first for mega), else its official artwork (PokeAPI), else the middle of
    one of its cards' art."""
    try:
        sources = [lambda: _first(rank(_search("evolveFrom", name), PORTRAIT_ERAS), disc_sprite)[1],
                   lambda: find_xy_mega(name)]
        for source in (sources[::-1] if mega else sources) + [lambda: pokeapi_sprite(name)]:
            pic = source()
            if pic is not None:
                return pic
        used, pic = _first(rank(_search("name", name), ART_ERAS), art_sprite)
    except cache.DownloadError as e:
        warnings.warn(f"portrait of {name}: TCGdex not reachable ({e})")
        return None
    if pic is None:
        warnings.warn(f"no card of {name!r} found for the evolution portrait; name a card in 'Portrait from card'")
        return None
    warnings.warn(f"no card portrait or artwork of {name} found: the portrait is the middle of {used}'s art")
    return pic


def smooth_disc(canvas):
    """The light disc evened out (the empty Stage sprite disc keeps faint lines of the figure taken off it), for a
    portrait that leaves some of it showing."""
    cx, cy, r = meframe.PORTRAIT_DISC
    x0, y0, size = round(cx - r) - 2, round(cy - r) - 2, round(2 * r) + 4
    region = canvas[y0:y0 + size, x0:x0 + size, :3]
    yy, xx = np.mgrid[y0:y0 + size, x0:x0 + size]
    inside = (np.hypot(xx - cx, yy - cy) < r - DISC_RIM).astype(np.float32)
    num = cv2.GaussianBlur(region * inside[..., None], (0, 0), DISC_SMOOTH)
    den = cv2.GaussianBlur(inside, (0, 0), DISC_SMOOTH)[..., None]
    region[:] = np.where(inside[..., None] > 0, num / np.maximum(den, 1e-3), region)


def put_portrait(canvas, rgb, alpha):
    """Centre a portrait sprite on the ring's disc (evened out first); recorded as 'portrait'."""
    smooth_disc(canvas)
    h, w = alpha.shape
    cx, cy, r = meframe.PORTRAIT_DISC
    x0, y0 = round(cx - w / 2), round(cy - h / 2)
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    a = (alpha * np.clip(r - 0.5 - np.hypot(xx - cx, yy - cy), 0, 1))[..., None]
    region = canvas[y0:y0 + h, x0:x0 + w, :3]
    region[:] = region * (1 - a) + rgb * a
    with layers.kind("portrait"):
        layers.record(a[..., 0], x0, y0)


def portrait(card, own=None):
    """The portrait for card: the named card's, else the card's own (own(): (rgb, alpha)) while it still evolves
    from what its own portrait shows, else found by name."""
    if card.get("portrait"):
        return from_card(card["portrait"])
    if own and card["evolves_from"] == card.get("portrait_of"):
        return own()
    return find(card["evolves_from"], mega=card["stage"] == MEGA)


# ---------------------------------------------------------------- text
@functools.lru_cache(maxsize=1)
def evolves_style():
    """dict(face, size, xscale, tracking, base) of the "Evolves from" line."""
    path = _font_file(EVOLVES_FONT)
    if path:
        return dict(face=path, **EVOLVES_FIT)
    warnings.warn(f"{EVOLVES_FONT} (Segoe UI Bold Italic) not found: the \"Evolves from\" line is set in Gill Sans, "
                  f"which looks wider than the print; copy the font into the fonts folder to fix")
    return EVOLVES_FALLBACK


def evolves_line(canvas, name, right=EVOLVES_RIGHT):
    """The "Evolves from" line, squeezed further to end left of `right`."""
    style = evolves_style()
    text = f"Evolves from {name}"
    width = ts.text_width(text, style["face"], style["size"], style["xscale"], style["tracking"])
    xscale = style["xscale"] * min(1.0, (right - EVOLVES_X) / width)
    with layers.kind("evolves"):
        ts.stamp(canvas, ts.coverage([dict(text=text, face=style["face"], size=style["size"], x=EVOLVES_X,
                                           base=style["base"], xscale=xscale, tracking=style["tracking"])]),
                 np.array(frame.DARK_INK, np.float32))


# ---------------------------------------------------------------- all together
def strip_end(card):
    """x where the card's "Evolves from" strip ends."""
    mega = card.get("family") == "ex" and card["name"].startswith("Mega ")
    return MEGA_STRIP_END if mega else STRIP[2]


def draw(canvas, card, parts, own=None, printed=False):
    """card["stage"]'s pieces `parts` (REGULAR, EX or a few of them); the disc gets the portrait and the strip the
    "Evolves from" line. Stage 0: the BASIC tab, when parts has the tab. printed: the card prints its own Stage
    pieces already (an SV / ME Stage card edited in place): its strip and disc stay (the caller clears the old line),
    only the line and the portrait are set again."""
    if not card["stage"]:
        if "tab" in parts:
            paste(canvas, basic_tab(), part_mask(("tab",)))
        return
    sprite = sprite_of(card["stage"])
    end = strip_end(card)
    where = part_mask(parts)
    if printed:
        where &= ~part_mask(("strip", "disc"))
    mega_in_place = card["stage"] == MEGA and "tab" in parts and shows_tab(canvas)
    if mega_in_place:
        where &= ~part_mask(("tab",))
    where[:, end:] &= ~_boxes()["strip"][:, end:]
    paste(canvas, sprite, where)
    if mega_in_place:
        set_mega(canvas)
    if "disc" in parts:
        pic = portrait(card, own)
        if printed:                   # the old picture off first: a cut-out figure would not cover it
            paste(canvas, sprite, part_mask(("disc",)))
        if pic is not None:
            put_portrait(canvas, *pic)
    if "strip" in parts:
        evolves_line(canvas, card["evolves_from"], min(EVOLVES_RIGHT, end - STRIP_PAD))
