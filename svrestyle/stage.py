"""The Stage pieces top left, as Scarlet & Violet and Mega Evolution print them: the STAGE tab, the silver ring with
the evolution portrait, the "Evolves from" strip, and on regular cards a silver swoosh in the name bar.

The pieces are meframe.stage_sprite's (lifted from me1 Kadabra and Alakazam). They sit at the same place on SV and
ME cards, regular and ex: checked on Kadabra and Alakazam (sv3.5, me1), Alakazam ex and Mega Venusaur ex. Every one
of those prints the same picture of the pre-evolution on the same light disc, so a portrait is lifted whole from a
card TCGdex lists as evolving from that Pokemon. A card id can be named instead: an SV / ME Stage card gives its
portrait, any other card the middle of its art window.

Besides Basic, Stage 1 and Stage 2 there is a custom stage, Mega (MEGA): the Stage 1 pieces with the tab reading MEGA.
"""
import functools
import os
import re
import warnings
from pathlib import Path
from urllib.parse import quote

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
# No print has a MEGA tab. Century Gothic Bold, widened and slanted, matches the BASIC / STAGE lettering (round G,
# pointed A, same weight; cap height and "STAGE" width measured on me1 Kadabra); set like it: dark fill, white rim,
# thin dark edge. Without Century Gothic, Gill Sans Bold.
MEGA_FONT = "GOTHICB.TTF"
MEGA_CAP, MEGA_XSCALE, MEGA_SHEAR = 11.5, 1.4, 0.32     # px, width factor, slant (x per y)
OLD_LETTERS = 0.65                # HSV value under which a tab pixel may belong to STAGE1's letters
LETTER_STANDS_OUT = 0.08          # how far (HSV value) letters and rim stand out from the tab around them
MEGA_LEFT, MEGA_BOTTOM = 28, 50                          # ink left x and last ink row, as STAGE on Kadabra
MEGA_RIM, MEGA_EDGE = 1.3, 0.7                           # px: white rim, then the thin dark edge round it
MEGA_INK = {"fill": (70, 70, 74), "rim": (255, 252, 255), "edge": (96, 96, 100)}   # sampled on Kadabra's tab
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


def _word(path, text, cap, xscale, shear, strokes):
    """Coverage (h x w, 1x) of text grown by each stroke width, slanted and widened, and the fill's ink box."""
    size = 4.0
    while True:                          # font size whose E is `cap` px tall
        box = ImageFont.truetype(path, round(size * WORD_SS)).getbbox("E")
        if (box[3] - box[1]) / WORD_SS >= cap or size > 60:
            break
        size += 0.25
    font = ImageFont.truetype(path, round(size * WORD_SS))
    w, h, base = int(font.getlength(text)) + 30 * WORD_SS, 40 * WORD_SS, 30 * WORD_SS
    out = []
    for stroke in strokes:
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((8 * WORD_SS, base), text, font=font, fill=255, anchor="ls",
                                 stroke_width=round(stroke * WORD_SS), stroke_fill=255)
        img = img.transform(img.size, Image.AFFINE, (1, -shear, shear * base, 0, 1, 0), Image.BICUBIC)
        out.append(np.asarray(img.resize((max(1, round(w * xscale / WORD_SS)), h // WORD_SS), Image.BOX),
                              np.float32) / 255)
    ys, xs = np.nonzero(out[0] > 0.5)
    return out, (xs.min(), ys.max())


def mega_tab():
    """RGBA 733x1024: the Stage 1 pieces with the tab reading MEGA (cached)."""
    out = cache.path("assets", f"me-v{meframe.ASSET_VERSION}", "mega-tab.png")
    if out.exists():
        return load_rgba(out)
    sprite = meframe.stage_sprite(1).copy()
    box = np.zeros((H, W), bool)
    x0, y0, x1, y1 = LETTER_BOX
    box[y0:y1, x0:x1] = True
    # the tab is opaque all over its lozenge (Kadabra's cut-out lets parts of the 1 through)
    _, _, val = hsv(sprite[..., :3])
    lozenge = fill_holes((sprite[..., 3] > 128) | (box & (val < OLD_LETTERS))) & box
    sprite[lozenge, 3] = 255
    # STAGE1 off: what stands out from the tab's own shading (letters, rim, edge) is filled from around it (TELEA:
    # the tab's bevel shades smoothly, LaMa would be overkill for 3 px strokes)
    v8 = np.clip(val * 255, 0, 255).astype(np.uint8)
    around = cv2.medianBlur(v8, 15).astype(np.float32) / 255
    old = dilate(lozenge & (np.abs(val - around) > LETTER_STANDS_OUT), 1) & lozenge
    rgb = np.clip(sprite[..., :3], 0, 255).astype(np.uint8)
    sprite[..., :3] = cv2.inpaint(rgb, old.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)
    path = _font_file(MEGA_FONT)
    if path is None:
        warnings.warn(f"{MEGA_FONT} (Century Gothic Bold) not found: the MEGA tab is set in Gill Sans Bold")
        path = f"{ts._font_dir}/{ts.BOLD}.otf"
    (fill, rim, edge), (left, bottom) = _word(path, "MEGA", MEGA_CAP, MEGA_XSCALE, MEGA_SHEAR,
                                              (0, MEGA_RIM, MEGA_RIM + MEGA_EDGE))
    dx, dy = MEGA_LEFT - left, MEGA_BOTTOM - bottom
    for layer, ink in ((edge, "edge"), (rim, "rim"), (fill, "fill")):
        a = np.zeros((H, W), np.float32)
        lh, lw = layer.shape
        a[dy:dy + lh, dx:dx + lw] = layer[:H - dy, :W - dx]
        a *= box
        sprite[..., :3] = sprite[..., :3] * (1 - a[..., None]) + np.array(MEGA_INK[ink], np.float32) * a[..., None]
    save_rgba(sprite, out)
    return sprite


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
    """TCGdex card briefs whose `field` equals name (cached)."""
    tcgdex_name = name.replace("’", "'")          # TCGdex writes a straight apostrophe (Farfetch'd)
    slug = re.sub(r"[^\w-]+", "_", tcgdex_name)
    return cache.fetch_json(f"{tcg.TCGDEX}/cards?{field}=eq:{quote(tcgdex_name)}",
                            cache.path("data", f"search-{field}-{slug}.json")) or []


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


def from_card(card_id):
    """(rgb, alpha) from a card the user named: its portrait if it is an SV / ME Stage card, else its art."""
    c = tcg.card(card_id)
    if card_id.startswith(PORTRAIT_ERAS) and c.get("stage") in ("Stage1", "Stage2"):
        return disc_sprite(card_id)
    return art_sprite(card_id)


def find(name):
    """(rgb, alpha) of the Pokemon called name, or None (with a warning)."""
    tried = 0
    try:
        for ids, sprite in ((rank(_search("evolveFrom", name), PORTRAIT_ERAS), disc_sprite),
                            (rank(_search("name", name), ART_ERAS), art_sprite)):
            for card_id in ids[:MAX_TRIES]:
                try:
                    out = sprite(card_id)
                except (cache.DownloadError, ValueError):
                    continue
                if sprite is art_sprite:
                    warnings.warn(f"no Scarlet & Violet / Mega Evolution card evolves from {name}: the portrait is "
                                  f"the middle of {card_id}'s art")
                return out
            tried += len(ids)
    except cache.DownloadError as e:
        warnings.warn(f"portrait of {name}: TCGdex not reachable ({e})")
        return None
    warnings.warn(f"no card of {name!r} found for the evolution portrait ({tried} tried); name a card in "
                  f"'Portrait from card'")
    return None


def put_portrait(canvas, rgb, alpha):
    """Centre a portrait sprite on the ring's disc; recorded as 'portrait'."""
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
    return find(card["evolves_from"])


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
    where[:, end:] &= ~_boxes()["strip"][:, end:]
    paste(canvas, sprite, where)
    if "disc" in parts:
        pic = portrait(card, own)
        if pic is not None:
            put_portrait(canvas, *pic)
        elif printed:                 # no picture found: an empty disc rather than the old one
            paste(canvas, sprite, part_mask(("disc",)))
    if "strip" in parts:
        evolves_line(canvas, card["evolves_from"], min(EVOLVES_RIGHT, end - STRIP_PAD))
