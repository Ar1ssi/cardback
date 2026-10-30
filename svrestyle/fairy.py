"""Fairy, which neither Scarlet & Violet nor Mega Evolution prints, in their look: its colours, symbol, icons and
frames, made from Psychic's.

Pokemon TCG Live gives every type a default Energy coin, Fairy's among them: one design in each type's colours, its
middle the same symbol artwork the cards' energy icons print (cosmetics: CN_TCGL_default_<type>-d). So:
- colour(rgb): Psychic's colours as Fairy's, the way the Psychic coin's turn into the Fairy coin's pixel for pixel
  (their symbols left out): in Oklab a lightness shift, a chroma gain and a hue turn, each a function of the Psychic
  colour's lightness, fitted per LIGHT_STEP of it. The coin's colours are its glossy disc's mid and dark tones; past
  them (a frame's pastel panel) its lightness and chroma changes run on too far (a hot pink, its mottling washed out),
  so from LIGHT_TAPER on the map keeps the colour's lightness and chroma and turns its hue alone. Greys (silver,
  white, the black of icons and ink) stay as they are.
- symbol: the Fairy coin's dark shape, filled (the coin shades it in 3-D; the cards print their symbols flat). An icon
  of radius r prints the coin's art at r / COIN_ART_R, its middle on the coin's (the Psychic and Grass coins' symbols
  land on their icons' within a pixel, on both icon sizes). In effect text the bare symbol prints in the text's ink,
  the art GLYPH_R px in radius at full body size, its middle GLYPH_AT from the gap's centre and the baseline (the
  Psychic, Grass and Darkness symbols lifted off the prints, lifted.py).
Every type's frame is one design in its colours, so Fairy's is Psychic's through colour(), with Fairy's icons: the
Mega Evolution icons (the Psychic icon's disc, its eye filled from the disc round it, through colour(), the symbol set
on it in the black the icons print their symbols in; meframe.icon), the regular frame (meframe.template), the
Scarlet & Violet ex name bar and panel (frame.py), the Mega ex frame (megaretype.py) and the Basic Energy (energy.py).
The Mega Evolution typed Special Energy (Telepathic Psychic Energy and its like) prints its symbol big on a glass
sphere: Fairy's is Psychic's sphere with the eye painted out and Fairy's symbol inked in the eye's own gloss, under the
print's glare (typed_energy).
Needs the TCG Live extract (the coins).
"""
import functools
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import cache, tcg
from .imaging import dilate, fill_holes, load_rgba, luminance, save_rgba

W, H = 733, 1024
ASSET_VERSION = "3"              # 3: the symbol's edge antialiased at any size
BASE = "Psychic"                 # the type whose frames Fairy's are made from
COIN = "CN_TCGL_default_{}-d"    # the default Energy coin's texture, per type (lower case)
COIN_C = (253.5, 253.5)          # px: the coin's middle on its texture
COIN_SYMBOL_R = 172              # px: the symbol lies inside this radius (the disc's dark rim lies past it)
COIN_ART_R = 164.0               # px of the coin's art an icon's radius covers
SYMBOL_LUM = (40, 95)            # coin luminance where it is all symbol / none
SYMBOL_DARK = 0.5                # coverage over which a coin pixel is the symbol's (its shading inside it filled)
SYMBOL_SEED = 0.95               # coverage a symbol's shape holds somewhere (the disc's shading never gets so dark)
SYMBOL_UP = 8                    # the symbol is set at this many times the size and averaged down
LIGHT_STEP = 0.05                # Oklab lightness per bin of the colour map
LIGHT_MIN = 200                  # pixel pairs a bin needs (others take the nearest bins')
GREY_C = (0.015, 0.045)          # Oklab chroma under which a colour stays as it is, over which it all turns
LIGHT_TAPER = (0.65, 0.85)       # Oklab lightness from which the map's lightness and chroma changes fade, to none
GAMUT_STEPS, GAMUT_SHRINK = 12, 0.9   # a colour turned out of sRGB loses chroma in these steps until it is in
SYMBOL_EDGE = 9                  # px round the coins' symbols left out of the colour map (their shading)
MATCH_REACH, MATCH_STEP = 0.15, 0.01   # the art radius Psychic's symbol is looked for at: this share either side
DISC_INK = 0.92                  # share of an icon's radius its symbol's ink lies in
EYE_GROW = (1.5, 0.03)           # px round the printed symbol taken off with it, and share of its art radius (its
                                 # soft glow on a big one)
DISC_IN = 1.0                    # px inside the icon's rim its disc is read from
FILL_SIGMAS = (0.06, 0.12, 0.25, 0.5)   # share of the art radius: the ground's colour about a pixel is read this wide
# The light ring round an icon's symbol (glow): read in steps of GLOW_STEP px (on an icon of art radius GLOW_SIZE; in
# proportion on bigger ones) out to GLOW_TAIL[1] of the art radius, against the disc from GLOW_TAIL[0] on (its own
# shading past the ring); a ring lifting the lightness less than GLOW_LIFT is none. Psychic's: +0.13..0.22 over 2.5 px.
GLOW_STEP, GLOW_SIZE, GLOW_TAIL, GLOW_LIFT, GLOW_MIN = 0.5, 26.0, (0.13, 0.21), 0.03, 8
ICON_MAX_R = 30                  # px: discs up to this radius are energy icons (glossy, a light ring round the symbol)
GLYPH_R = 15.1                   # px: the symbol's art radius in effect text at full body size
GLYPH_AT = (-0.3, -8.9)          # px: its middle from the gap's centre, and from the baseline
# Psychic's Mega Evolution Basic Energy (mee-005) prints its symbol four times: bare in the "Basic {P} Energy" line, in
# the icons top right and in the footer, and big on the ball. Centre x, y and art radius about where they are (Hough;
# swap matches them; the bare one by matching along the line), and the disc's radius (None: bare on the name bar).
ENERGY_SYMBOLS = [(185.0, 100.0, 23.0, None), (669.5, 104.5, 26.6, 26.6), (676.5, 976.5, 13.6, 13.6),
                  (364.5, 542.5, 160.0, 176.5)]
# The gold Hyper Rare Basic Energy (sv03.5-207 and the other types' alike): only its art window and the header's swoosh
# print the type's colour, the rest is gold (x0, y0, x1, y1; the window measured where Psychic's and Grass's prints
# differ). The swoosh's purple is told from the gold by blue over green: (offset, ramp) in RGB levels.
GOLD_ART = (28, 140, 705, 949)
GOLD_SWOOSH = (295, 20, 372, 76)
SWOOSH_BLUE = (10, 50)
# Psychic's Mega Evolution typed Special Energy (me03-088 Telepathic {P} Energy) prints its symbol bare in its name and
# big on the glass sphere of its art (centre x, y and art radius, as match finds them). The eight typed prints share one
# sphere (TYPED_SPHERE: centre and radius inside its dark outline) and its glare; only the symbol and colours differ.
TYPED_ENERGY = "me03-088"
TYPED_NAME_EYE = (301.4, 104.1, 26.45)
TYPED_EYE = (362.55, 393.2, 245.0)
TYPED_SPHERE = (366.0, 397.5, 214.0)
# The eye's ink on the sphere: black, or teal where the sphere's gloss lights it (Oklab hue), against the sphere's
# purple; looked for within EYE_AREA px of the matched eye.
INK_BLACK, INK_HUE, SPHERE_HUE, INK_C, EYE_AREA = 0.16, 220.0, 308.0, 0.02, 8
# The ink's gloss as a smooth field, read INK_SIGMA px round each point (INK_FAR where the eye lies farther off than
# INK_NEAR of the near reading's weight), from the eye's ink INK_IN px inside its edge.
INK_SIGMA, INK_FAR, INK_NEAR, INK_IN = 22.0, 60.0, 0.15, 2
# The glare by the eye's lower right: its core (Oklab L over GLARE_L within GLARE_FIND px of GLARE_AT) is left out of
# the ink's field; it is lifted off the print (how far the print goes from what lies under it toward the core's own
# colour), softened GLARE_SOFT px, out to GLARE_R px from its middle, faded from CORE_R.
GLARE_AT, GLARE_FIND, GLARE_L, CORE_R, GLARE_R, GLARE_SOFT = (483, 493), 40, 0.95, 23, 30, 1.8

# Oklab (Bjorn Ottosson): linear sRGB -> LMS -> Lab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _asset(name):
    return cache.path("assets", f"fairy-v{ASSET_VERSION}", name)


_build = threading.RLock()


def _locked(build):
    """An asset builder that runs one call at a time (the GUI serves requests on threads: a page's icons and a render
    can ask for the same new asset at once)."""
    @functools.wraps(build)
    def run(*args, **kwargs):
        with _build:
            return build(*args, **kwargs)
    return run


def tag(type_name):
    """What the cached assets made from this module carry in their names, so they are made again when it changes:
    "-f<version>" for Fairy, nothing for a printed type."""
    return f"-f{ASSET_VERSION}" if type_name == "Fairy" else ""


def available():
    """Whether Fairy can be made: the TCG Live extract holds the coins."""
    try:
        coin(BASE.lower())
        coin("fairy")
        return True
    except (LookupError, OSError, ImportError):
        return False


@functools.lru_cache(maxsize=None)
def coin(type_name):
    """The type's default Energy coin texture (RGB float32, 512x512): its decoded PNG in the extract, else read out of
    its bundle (UnityPy)."""
    if not tcg.TCGLIVE:
        raise LookupError("Fairy needs a Pokemon TCG Live extract (its symbol and colours come from the game's Energy "
                          "coins): pass --tcglive DIR or set SVRESTYLE_TCGLIVE")
    name = COIN.format(type_name)
    folder = f"cn_tcgl-default_energy-{type_name}_mtl"
    png = Path(tcg.TCGLIVE, "cosmetics", "png", "coins", folder, f"_MainTex__{name}.png")
    if png.exists():
        img = Image.open(png)
    else:
        import UnityPy
        bundle = Path(tcg.TCGLIVE, "cosmetics", "bundles_deps", f"cn_tcgl-default_energy-{type_name}")
        if not bundle.exists():
            raise LookupError(f"no {type_name} Energy coin in the TCG Live extract ({png})")
        img = next((o.read().image for o in UnityPy.load(str(bundle)).objects
                    if o.type.name == "Texture2D" and o.peek_name() == name), None)
        if img is None:
            raise LookupError(f"no texture {name} in {bundle}")
    return np.asarray(img.convert("RGB"), np.float32)


# ---------------------------------------------------------------- colours
def oklab(rgb):
    c = np.clip(np.asarray(rgb, np.float64) / 255, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def from_oklab(lab):
    lin = np.clip(((lab @ np.linalg.inv(_M2).T) ** 3) @ np.linalg.inv(_M1).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255


@functools.lru_cache(maxsize=None)
@_locked
def colour_map():
    """(bin middles of Psychic lightness, per bin: lightness shift, log chroma gain, hue turn in degrees)."""
    out = _asset("colours.npz")
    if not out.exists():
        p, f = oklab(coin(BASE.lower())), oklab(coin("fairy"))
        symbols = (p[..., 0] < 0.45) | (f[..., 0] < 0.45)
        clear = ~dilate(symbols, SYMBOL_EDGE)
        cp, cf = np.hypot(p[..., 1], p[..., 2]), np.hypot(f[..., 1], f[..., 2])
        pairs = clear & (cp > GREY_C[1])
        light = p[pairs, 0]
        shift = f[pairs, 0] - light
        gain = np.log(cf[pairs] / cp[pairs])
        turn = np.degrees(np.arctan2(f[pairs, 2], f[pairs, 1]) - np.arctan2(p[pairs, 2], p[pairs, 1]))
        turn = (turn + 180) % 360 - 180
        edges = np.arange(0, 1 + 1e-9, LIGHT_STEP)
        middles = (edges[:-1] + edges[1:]) / 2
        rows = np.full((len(middles), 3), np.nan)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (light >= lo) & (light < hi)
            if m.sum() >= LIGHT_MIN:
                rows[i] = np.median(shift[m]), np.median(gain[m]), np.median(turn[m])
        have = ~np.isnan(rows[:, 0])
        table = np.stack([np.interp(middles, middles[have], rows[have, k]) for k in range(3)], -1)
        np.savez(out, middles=middles, table=table)
    d = np.load(out)
    return d["middles"], d["table"]


def colour(rgb):
    """Psychic's colours (RGB, any shape ending in 3) as Fairy's. float32"""
    middles, table = colour_map()
    lab = oklab(rgb)
    light, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma, hue = np.hypot(a, b), np.arctan2(b, a)
    lo, hi = GREY_C
    w = np.clip((chroma - lo) / (hi - lo), 0, 1)                 # greys stay
    lo, hi = LIGHT_TAPER
    k = w * np.clip((hi - light) / (hi - lo), 0, 1)              # pastels: the hue turn alone
    light2 = light + k * np.interp(light, middles, table[:, 0])
    chroma2 = chroma * np.exp(k * np.interp(light, middles, table[:, 1]))
    hue2 = hue + np.radians(w * np.interp(light, middles, table[:, 2]))
    for _ in range(GAMUT_STEPS):                # dark violets turned pink leave sRGB: less chroma, not clipped
        lin = ((np.stack([light2, chroma2 * np.cos(hue2), chroma2 * np.sin(hue2)], -1) @ np.linalg.inv(_M2).T) ** 3
               @ np.linalg.inv(_M1).T)
        out = ((lin < -1e-4) | (lin > 1 + 1e-4)).any(-1)
        if not out.any():
            break
        chroma2 = np.where(out, chroma2 * GAMUT_SHRINK, chroma2)
    out = np.stack([light2, chroma2 * np.cos(hue2), chroma2 * np.sin(hue2)], -1)
    return from_oklab(out).astype(np.float32)


# ---------------------------------------------------------------- the symbol
@functools.lru_cache(maxsize=None)
def coin_symbol(type_name="fairy", solid=True):
    """The type's symbol on its coin: coverage (512x512 float32). Its shapes are the dark ones holding pixels all of
    symbol's darkness (the disc's own shading, darker towards its rim, is not); solid: filled where the coin shades
    them lighter, a pixel inside their edge, which keeps the coin's own antialiasing (a symbol to set; else as the
    coin has it, to be matched with a print: an eye's white stays out)."""
    lum = luminance(coin(type_name))
    yy, xx = np.mgrid[:lum.shape[0], :lum.shape[1]]
    inside = np.hypot(xx - COIN_C[0], yy - COIN_C[1]) < COIN_SYMBOL_R
    lo, hi = SYMBOL_LUM
    cover = np.clip((hi - lum) / (hi - lo), 0, 1) * inside
    _, labels = cv2.connectedComponents((cover > 0.05).astype(np.uint8))
    shapes = np.setdiff1d(np.unique(labels[cover >= SYMBOL_SEED]), [0])
    cover = cover * np.isin(labels, shapes)
    if solid:
        core = cv2.morphologyEx((cover > SYMBOL_DARK).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        filled = cv2.erode(fill_holes(core > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        cover = np.maximum(cover, filled)
    return cover.astype(np.float32)


def symbol(shape, cx, cy, r, type_name="fairy", solid=True):
    """Coverage (shape: (h, w)) of the symbol an icon of radius r centred on (cx, cy) prints (coin_symbol). solid: its
    outline cut where the coin's coverage, interpolated SYMBOL_UP times finer, crosses half, and averaged down, so an
    edge is antialiased alike at any size (the coin's pixels, magnified on a big symbol, would show as steps)."""
    up = SYMBOL_UP
    s = r / COIN_ART_R * up
    m = np.float32([[s, 0, (cx + 0.5) * up - 0.5 - COIN_C[0] * s], [0, s, (cy + 0.5) * up - 0.5 - COIN_C[1] * s]])
    big = cv2.warpAffine(coin_symbol(type_name, solid), m, (shape[1] * up, shape[0] * up),
                         flags=cv2.INTER_CUBIC if solid else cv2.INTER_LINEAR)
    if solid:
        big = (big > 0.5).astype(np.float32)
    return cv2.resize(big, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def glyph(up, scale=1.0):
    """The symbol in effect text: (coverage at up x, its top-left from the gap's centre and the baseline, up x px),
    as lifted.symbol places its glyphs; scale: the body text's size over the full size."""
    r = GLYPH_R * scale * up
    half = int(np.ceil(r * COIN_SYMBOL_R / COIN_ART_R)) + 2
    shape = symbol((2 * half, 2 * half), half - 0.5, half - 0.5, r)
    return shape, GLYPH_AT[0] * scale * up - (half - 0.5), GLYPH_AT[1] * scale * up - (half - 0.5)


# ---------------------------------------------------------------- icons and frames
def match(crop, cx, cy, r, inside=None):
    """Where Psychic's symbol prints on crop, about (cx, cy) at about art radius r: (centre x, y, art radius) where the
    Psychic coin's symbol lands best on the print's dark ink (normalized correlation; the radius to MATCH_STEP of it,
    the centre to a fraction of a pixel). inside: bool mask of where ink is the symbol's (an icon's disc)."""
    lo, hi = SYMBOL_LUM
    ink = np.clip((hi - luminance(crop)) / (hi - lo), 0, 1).astype(np.float32)
    if inside is not None:
        ink *= inside
    best = None
    for k in np.arange(1 - MATCH_REACH, 1 + MATCH_REACH + 1e-9, MATCH_STEP):
        half = int(np.ceil(r * k * COIN_SYMBOL_R / COIN_ART_R)) + 2
        if 2 * half + 1 >= min(crop.shape[:2]):
            continue
        tmpl = symbol((2 * half + 1, 2 * half + 1), half, half, r * k, BASE.lower(), solid=False)
        score = cv2.matchTemplate(ink, tmpl, cv2.TM_CCORR_NORMED)
        _, peak, _, (bx, by) = cv2.minMaxLoc(score)
        if best is None or peak > best[0]:
            best = (peak, r * k, half, bx, by, score)
    _, art_r, half, bx, by, score = best

    def vertex(a, b, c):                        # the peak of a parabola through three samples, from the middle one
        d = a - 2 * b + c
        return 0.0 if d >= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    fx = vertex(*score[by, bx - 1:bx + 2]) if 0 < bx < score.shape[1] - 1 else 0.0
    fy = vertex(*score[by - 1:by + 2, bx]) if 0 < by < score.shape[0] - 1 else 0.0
    return bx + fx + half, by + fy + half, art_r


def glow(crop, printed, disc, art_r):
    """The light ring an icon prints round its symbol (Psychic's eye, Grass's leaf...): (distances out from the
    symbol, Oklab lightness and chroma gain at each over the disc just past it), read off the print, the disc's own
    shading (lighter in the middle) taken out; None when there is none."""
    d = cv2.distanceTransform((~printed).astype(np.uint8), cv2.DIST_L2, 5)
    lab = oklab(crop)
    step = max(GLOW_STEP, art_r * GLOW_STEP / GLOW_SIZE)
    rows = []
    for at in np.arange(step, GLOW_TAIL[1] * art_r, step):
        band = disc & ~printed & (np.abs(d - at) < step / 2)
        if band.sum() >= GLOW_MIN:
            rows.append((at, np.median(lab[band, 0]), np.median(np.hypot(lab[band, 1], lab[band, 2]))))
    if not rows:
        return None
    at, light, chroma = map(np.array, zip(*rows))
    tail = (at >= GLOW_TAIL[0] * art_r) & (at <= GLOW_TAIL[1] * art_r)
    if not tail.any():
        return None
    lift = np.clip(light - np.median(light[tail]), 0, None)
    if lift.max() < GLOW_LIFT:
        return None
    return at, lift, np.where(lift > 0, chroma / max(float(np.median(chroma[tail])), 1e-3), 1.0)


def swap(crop, cx, cy, r, disc_r=None, recolour=True):
    """Psychic's symbol on crop made Fairy's: found about (cx, cy) at about art radius r (match), filled from what is
    round it (an icon's disc, disc_r px round (cx, cy); else the light ground the bare symbol prints on), through
    colour(), and the Fairy symbol set where Psychic's was, in the print's ink through colour(). On an icon (a disc
    up to ICON_MAX_R px) the light ring round Psychic's symbol is taken off with it and set again round Fairy's
    (glow), and the disc keeps its own light and shade: the colour map's lightness change, fitted on the coins'
    flatter discs, would flatten its gloss. A Basic Energy's ball prints its symbol flat, and takes the map whole.
    recolour: False for a symbol on ground that is no type's colour (a gold title line): filled and inked as it is."""
    yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
    inside = np.hypot(xx - cx, yy - cy) < (disc_r * DISC_INK if disc_r else r * COIN_SYMBOL_R / COIN_ART_R + 2)
    cx, cy, art_r = match(crop, cx, cy, r, inside)
    lo, hi = SYMBOL_LUM
    ink = np.clip((hi - luminance(crop)) / (hi - lo), 0, 1)
    shape = symbol(crop.shape[:2], cx, cy, art_r, BASE.lower(), solid=False) > 0.05
    # the print's own shapes the coin's lands on, whole: a bare symbol's glyph is drawn a little otherwise (the
    # name line's eye has a rounder lid)
    _, labels = cv2.connectedComponents(((ink > 0.5) & inside).astype(np.uint8))
    printed = fill_holes(shape | np.isin(labels, np.setdiff1d(np.unique(labels[shape]), [0])))
    disc = (np.hypot(xx - cx, yy - cy) < disc_r - DISC_IN) if disc_r else None
    icon = bool(disc_r) and disc_r <= ICON_MAX_R
    ring = glow(crop, printed, disc, art_r) if icon else None
    grow = GLOW_TAIL[0] * art_r if disc_r else max(EYE_GROW[0], EYE_GROW[1] * art_r)
    hole = dilate(printed, grow)
    ground = ~hole & (disc if disc_r else ink < 0.1)
    filled, left = crop.copy(), hole.copy()
    for k in FILL_SIGMAS:                       # the ground's colour about each pixel, from the nearest that reach it
        sigma = k * art_r
        num = cv2.GaussianBlur(crop * ground[..., None].astype(np.float32), (0, 0), sigma)
        den = cv2.GaussianBlur(ground.astype(np.float32), (0, 0), sigma)
        take = left & ((den > 0.3) | (k == FILL_SIGMAS[-1]))
        filled[take] = (num / np.maximum(den, 1e-6)[..., None])[take]
        left &= ~take
    colour_ink = np.median(crop[(ink > 0.9) & printed], axis=0)[None]
    colour_ink = (colour(colour_ink) if recolour else colour_ink)[0]
    out = colour(filled) if recolour else filled
    if icon:
        lab, own = oklab(out), oklab(filled)            # the disc's own light and shade, at the map's mean shift
        lab[..., 0] = np.where(disc, own[..., 0] + np.median(lab[disc, 0] - own[disc, 0]), lab[..., 0])
        if ring is not None:
            lab = _glowed(lab, disc, (cx, cy, art_r), ring)
        out = from_oklab(lab).astype(np.float32)
    cover = symbol(crop.shape[:2], cx, cy, art_r)[..., None]
    return out * (1 - cover) + colour_ink * cover


def _glowed(lab, disc, where, ring):
    """Oklab lab with the light ring (glow) round the Fairy symbol at where = (cx, cy, art radius): each pixel lifted
    by the ring's lightness and chroma at its distance out from the symbol, worked out SYMBOL_UP times finer."""
    cx, cy, art_r = where
    up = SYMBOL_UP
    h, w = disc.shape
    solid = symbol((h * up, w * up), (cx + 0.5) * up - 0.5, (cy + 0.5) * up - 0.5, art_r * up) > 0.5
    d = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 5) / up
    at, lift, gain = ring
    small = lambda a: cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    # over the part of each pixel the symbol leaves open: the symbol is set over it, its edge mixing with the ring
    open_share = np.maximum(small(~solid), 1e-3)
    lift = small(np.where(solid, 0, np.interp(d, at, lift, left=lift[0], right=0))) / open_share
    gain = small(np.where(solid, 0, np.interp(d, at, gain, left=gain[0], right=1))) / open_share
    out = lab.copy()
    out[..., 0] += lift * disc
    out[..., 1:] *= np.where(disc, gain, 1)[..., None]
    return out


@functools.lru_cache(maxsize=None)
@_locked
def icon_scan(card_id, cx, cy, r):
    """card_id's scan (a Psychic card) with its icon at (cx, cy), radius r, made Fairy's: what meframe.icon cuts Fairy's
    icons from. RGBA float32"""
    out = _asset(f"icon-{card_id}-{cx:.2f}-{cy:.2f}-{r:.2f}.png")
    if out.exists():
        return load_rgba(out)
    scan = tcg.scan(card_id).copy()
    x0, y0 = int(cx - r - 4), int(cy - r - 4)
    x1, y1 = int(cx + r + 5), int(cy + r + 5)
    scan[y0:y1, x0:x1, :3] = swap(scan[y0:y1, x0:x1, :3], cx - x0, cy - y0, r, disc_r=r)
    save_rgba(scan, out)
    return scan


def basic_energy(scan):
    """Psychic's Mega Evolution Basic Energy (scan: mee-005, RGBA) as Fairy's: through colour(), each of its symbols
    made Fairy's (swap: ENERGY_SYMBOLS)."""
    out = scan.copy()
    out[..., :3] = colour(scan[..., :3])
    for cx, cy, r, disc_r in ENERGY_SYMBOLS:
        reach = (disc_r or r * COIN_SYMBOL_R / COIN_ART_R) + 6
        x0, y0, x1, y1 = int(cx - reach), int(cy - reach), int(cx + reach) + 1, int(cy + reach) + 1
        out[y0:y1, x0:x1, :3] = swap(scan[y0:y1, x0:x1, :3], cx - x0, cy - y0, r, disc_r)
    return out


def gold_energy(scan):
    """Psychic's gold Hyper Rare Basic Energy (scan: sv03.5-207, RGBA) as Fairy's: its art window and the header's
    swoosh through colour(), the gold round them as it is (it is no type's colour); each symbol made Fairy's (swap:
    ENERGY_SYMBOLS, the gold print's are where the Mega Evolution one's are): the title line's on its gold as it is,
    the icons' discs in Fairy's colours inside their rims only."""
    out = scan.copy()
    rgb = scan[..., :3]
    x0, y0, x1, y1 = GOLD_ART
    share = np.zeros(rgb.shape[:2], np.float32)
    share[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 0.5                 # the window's edge: a pixel of blend with the gold
    share[y0:y1, x0:x1] = 1
    x0, y0, x1, y1 = GOLD_SWOOSH
    box = rgb[y0:y1, x0:x1]
    share[y0:y1, x0:x1] = np.clip((box[..., 2] - box[..., 1] - SWOOSH_BLUE[0]) / SWOOSH_BLUE[1], 0, 1)
    out[..., :3] = rgb + share[..., None] * (colour(rgb) - rgb)
    for cx, cy, r, disc_r in ENERGY_SYMBOLS:
        reach = (disc_r or r * COIN_SYMBOL_R / COIN_ART_R) + 6
        x0, y0, x1, y1 = int(cx - reach), int(cy - reach), int(cx + reach) + 1, int(cy + reach) + 1
        crop = rgb[y0:y1, x0:x1]
        new = swap(crop, cx - x0, cy - y0, r, disc_r, recolour=bool(disc_r))
        if disc_r and disc_r <= ICON_MAX_R:        # an icon on the gold: its disc and rim, not the gold round it
            yy, xx = np.mgrid[y0:y1, x0:x1]
            a = np.clip(disc_r + 0.5 - np.hypot(xx - cx, yy - cy), 0, 1)[..., None]
            new = a * new + (1 - a) * crop
        out[y0:y1, x0:x1, :3] = new
    return out


def _linear(rgb):
    c = np.clip(np.asarray(rgb, np.float64) / 255, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(lin):
    lin = np.clip(lin, 0, 1)
    return (np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255).astype(np.float32)


def _spread(values, known, sigma):
    """values (HxWx3) where known, spread sigma px round (normalized convolution). -> (field, its weight)"""
    w = known.astype(np.float32)
    num = cv2.GaussianBlur((values * w[..., None]).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None], den


def _typed_sphere(scan):
    """The typed print's sphere (scan: TYPED_ENERGY's, RGBA) with Fairy's symbol for Psychic's eye, still in Psychic's
    colours: the eye (its ink told from the sphere by hue) painted out by LaMa, Fairy's symbol inked with the eye's own
    gloss (a smooth field of the eye's ink: black on the left, lit teal toward the glare), and the print's glare lit
    again over it. In linear light. RGB float32"""
    from . import inpaint
    rgb = scan[..., :3].astype(np.float32)
    lab = oklab(rgb)
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    off = lambda h: np.abs((hue - h + 180) % 360 - 180)
    teal = (np.hypot(lab[..., 1], lab[..., 2]) > INK_C) & (off(INK_HUE) < off(SPHERE_HUE))
    near = dilate(symbol(rgb.shape[:2], *TYPED_EYE, BASE.lower()) > 0.5, EYE_AREA)
    eye = ((lab[..., 0] < INK_BLACK) | teal) & near
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    found = (np.hypot(xx - GLARE_AT[0], yy - GLARE_AT[1]) < GLARE_FIND) & (lab[..., 0] > GLARE_L)
    gd = np.hypot(xx - xx[found].mean(), yy - yy[found].mean())
    surface = _linear(inpaint.inpaint(scan, dilate(eye, 3) | (gd < GLARE_R))[..., :3])
    obs = _linear(rgb)
    ink = cv2.erode(eye.astype(np.uint8), np.ones((2 * INK_IN + 1,) * 2, np.uint8)) > 0
    known = ink & (gd >= CORE_R)
    near_field, weight = _spread(obs, known, INK_SIGMA)
    far_field, _ = _spread(obs, known, INK_FAR)
    w = np.clip(weight / INK_NEAR, 0, 1)[..., None]
    gloss = near_field * w + far_field * (1 - w)
    light = obs[found].mean(0)                          # the glare's own colour, a bluish white
    luma = np.array([0.2126, 0.7152, 0.0722])
    under = np.where(ink[..., None], gloss, surface) @ luma
    lift = np.clip((obs @ luma - under) / np.maximum(light @ luma - under, 1e-3), 0, 1).astype(np.float32)
    lift = cv2.GaussianBlur(lift, (0, 0), GLARE_SOFT) * np.clip((GLARE_R - gd) / (GLARE_R - CORE_R), 0, 1)
    cover = symbol(rgb.shape[:2], *TYPED_EYE)[..., None]
    # the symbol's edge mixed with the sphere as the print's are, in sRGB (in linear light a half-covered pixel takes
    # mostly the sphere's colour, and the edge shows its steps)
    inked = _linear(_srgb(surface) * (1 - cover) + _srgb(gloss) * cover)
    lit = inked * (1 - lift[..., None]) + light * lift[..., None]
    cx, cy, r = TYPED_SPHERE
    return np.where((np.hypot(xx - cx, yy - cy) < r)[..., None], _srgb(lit), rgb)


@_locked
def typed_energy():
    """Psychic's Mega Evolution typed Special Energy (TYPED_ENERGY, Telepathic Psychic Energy) as Fairy's: its sphere
    with Fairy's symbol (_typed_sphere) through colour(), and the symbol in its name made Fairy's (swap). Its name,
    text, icons and footer are set again for each card (energy.py). Cached. RGBA float32"""
    out_path = _asset("typed-energy.png")
    if out_path.exists():
        return load_rgba(out_path)
    scan = tcg.scan(TYPED_ENERGY)
    out = scan.copy()
    out[..., :3] = colour(_typed_sphere(scan))
    cx, cy, r = TYPED_NAME_EYE
    reach = r * COIN_SYMBOL_R / COIN_ART_R + 6
    x0, y0, x1, y1 = int(cx - reach), int(cy - reach), int(cx + reach) + 1, int(cy + reach) + 1
    out[y0:y1, x0:x1, :3] = swap(scan[y0:y1, x0:x1, :3], cx - x0, cy - y0, r)
    save_rgba(out, out_path)
    return out


def template(base):
    """The Mega Evolution regular frame of Fairy: Psychic's (base: meframe.template's RGBA) through colour(), with
    Fairy's name-bar icon. RGBA float32"""
    from . import meframe
    out = base.copy()
    out[..., :3] = colour(base[..., :3])
    (crop, alpha, scx, scy, (ox, oy)), _ = meframe.icon("Fairy", "name")
    h, w = alpha.shape
    region = out[oy:oy + h, ox:ox + w, :3]
    region[:] = region * (1 - alpha[..., None]) + crop * alpha[..., None]
    return out
