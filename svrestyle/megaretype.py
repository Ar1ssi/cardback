"""Change the type of a Mega Evolution ex on its own print (Double Rare, Ultra Rare, Special Illustration Rare, Mega
Hyper Rare), and of a Scarlet & Violet ex (non-Mega ex of the sv sets and of Mega Evolution's, Tera ex left out) the
same way: its frame is the one the Mega ex frame was drawn from (family SV, below).

A Mega ex Double Rare prints its type in four places, measured on every Double Rare of me01 to me05 and the promos
that share their frame (SOURCES), with their TCG Live white plates and etches (so a type change needs the extract):
- The name bar: opaque, one gradient per type, the same on each of the type's cards (types_bar fits it on the cards
  whose bars agree: me04-015's and some me01 Basics' stray); the name and ex logo are yellow on every type, the HP dark
  (white on Darkness). The card's own print is kept and the bar under it moved from the card's own bar (fitted smooth
  from its print-free pixels) to the new type's. A pixel's share of bar against the frame's silver pieces (tab, ring,
  strip, swoosh) is the slope of the layout's references' colours on their bars (bar_alpha), and along the pieces'
  edges the card's own (card_alpha): prints sit a fraction of a pixel apart.
- The text panel: the type's colour multiplied over the faded art. Its top envelope is the same colour on every card
  of a type, so a change there is a ratio per channel (new = old x tint_new / tint_old), the words re-inked (white on
  Darkness, dark elsewhere). Darkness prints no red under its panel: the faded art's red is read from its green and
  blue (faded_table, learnt on every other type's panels; it misses the print by about 13 levels, and nothing on the
  card holds the art's own red: the art above the panel shows other colours than under it). The panel's soft edge
  round what keeps its colours (icons, badge, splashes) takes its share of the change (edge_share).
- The glitter frame (the strip left of the panel, the weakness row's band, the footer behind its boxes, a sliver left
  of the portrait ring): the type's own glitter texture, the same pixels on each of its cards. A pixel in the old
  type's glitter colours takes the new type's texture where one of its cards shows it, else the new type's colour at
  the same luminance quantile.
- The name-bar icon: the same artwork as the Mega Evolution regular cards' (meframe.icon), from a card of the type.
Splashes, sparks and the Pokemon reaching over the frame are kept: the white plate tells them from the panel (its flat
value) and the glitter's colours from the glitter. Ultra Rare and Special Illustration Rare frames carry no type colour
(white lettering over the art), nor does a Mega Hyper Rare's gold: their icon alone changes (its disc is the same art
on every rarity), and a Mega Hyper Rare's etch, which carries the icon's symbol. Fairy, which no Mega ex prints, is
Psychic's frame in Fairy's colours (fairy.py).

The holofoil follows (plate): TCG Live's plates cut white lettering out (Darkness) and run the foil through dark, so a
card going to or from Darkness has its re-inked words cut out of the white plate, or closed.

A Scarlet & Violet ex Double Rare (SV, measured on its Double Rares of sv01 to sv10.5 and the non-Mega ex of me02 to
me05, SV_SOURCES) has the same name bar (the icon at the same place, the same artwork), and the same panel, the type's
colour multiplied over the faded art (Darkness again without red), but no glitter: below the panel, round the weakness
row, the rule box and the footer, and in the strips beside it, a band of the type's colour runs to the card's foot,
with the card's art reaching over it, no foil. What the old type's colour explains there moves to the new type's
(band_field); the frame's pieces on it, the same on every type (band_fixed), print and art keep theirs. Its name, HP
and weakness row words are dark, white on Darkness; its ex logo, and its footer (dark with a white keyline), are the
same on every type. Its
Ultra Rare, Special Illustration Rare, Hyper Rare (gold) and Shiny Ultra Rare print the same icon over a frame without
the type's colour: the icon alone changes (their etch rings it with plain circles, no symbol). Tera ex (their rule in
a silver box under the name bar, tera) and Black White Rares (a monochrome icon) are left out.
"""
import functools
import threading
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import cache, compose, foil, frame, meframe, tcg
from .imaging import dilate, drop_small, fill_holes, load_rgba, luminance, put_sprite, save_rgba

W, H = 733, 1024
ASSET_VERSION = "1"
# Every Mega ex Double Rare of me01..me05 in the TCG Live extract, by type, and the promos printed on their frame
# (those TCGdex has: it lacks mep-089..091).
SOURCES = {
    "Grass": ["me01-003", "me02-004", "me02.5-010", "mep-013", "mep-034"],
    "Fire": ["me02-013", "me02.5-022", "me01-022", "me02.5-031", "me04-015", "me05-008", "mep-029", "mep-030",
             "mep-035"],
    "Water": ["me01-036", "me02.5-043", "me02.5-047", "me03-021", "me04-022", "mep-036", "mep-081"],
    "Lightning": ["me01-050", "me02.5-061", "me05-027"],
    "Psychic": ["me01-060", "me02-041", "me02.5-089", "me03-031", "me04-035", "me05-031", "me05-038", "mep-072"],
    "Fighting": ["me01-077", "me02.5-113", "me02.5-116", "me03-047", "me04-048", "mep-012", "mep-071"],
    "Darkness": ["me01-086", "me02-056", "me02-061", "me02.5-125", "me02.5-135", "me05-048", "mep-073"],
    "Metal": ["me01-094", "me03-055", "me05-065"],
    "Dragon": ["me01-100", "me02.5-152", "me04-065", "mep-011"],
    "Colorless": ["me01-104", "me02-084", "me02.5-172", "mep-025"],
}
TYPES = tuple(SOURCES) + ("Fairy",)
DERIVED = {"Fairy": "Psychic"}            # a type no Mega ex prints: this type's frame in its colours (fairy.colour)
# Frame by TCGdex rarity: "dr" the Double Rare's (type-coloured bar, panel, glitter), "full" Ultra Rare / Special
# Illustration Rare (white lettering over the art), "gold" Mega Hyper Rare. A promo is "dr" when its bar is the type's.
FAMILY = {"Double rare": "dr", "Ultra Rare": "full", "Special illustration rare": "full", "Mega Hyper Rare": "gold"}

PANEL = (47, 517, 704, 873)
GLITTER = [(30, 505, 42, 873), (30, 879, 704, 996), (24, 56, 44, 100)]
BAR = (0, 0, W, 100)                       # name bar rows (the type's colour lives here)
# The card's own print on the glitter: footer (illustrator, mark, set box, number, rarity) and the weakness row's
# words and values. Glitter colours are learnt outside these; the footer keeps its print on every type.
FOOT_PRINT = (30, 915, 262, 990)
ROW_TEXT = (36, 880, 700, 911)
NAME_BOX, HP_BOX = (136, 26, 470, 96), (505, 22, 655, 98)
PANEL_SAMPLE = (100, 600, 650, 850)        # x0, y0, x1, y1: the plate's flat panel value is the mode here
QUANT = 64
LUT_STEP = 8                               # RGB levels per bin of the glitter colour table
GLIT_NEAR, GLIT_RAMP = 14, 22              # RGB distance from the old type's glitter colours: all glitter, none
GLIT_PLATE = 60                            # white plate under this: no foil, the glitter
WORD_CUT = 100          # plate value under which print is cut out of it (Darkness's white words: ~35)
EDGE_PX = 2             # px: the panel's soft edge round what keeps its colours
RING_PX = 3             # px: the ring round cut-outs whose pixels are read as blends with them
MIX_OFF = 18            # RGB a ring pixel may lie off a blend (or a shade of the panel) and be read as one
SHADE_MIN = 0.1         # least shade of the panel read as its shadow (darker is the cut-out's own)
EDGE_SPAN = 15          # plate levels between the panel's value and a cut-out's past which the edge is read off it
PLATE_RAMP = 22         # plate levels off the panel's value over which a pixel's share of panel falls to none
                        # (splashes over the panel sit ~26..36 off it; glows over it in between)
LOOKS_NEAR, LOOKS_RAMP = 6, 20   # RGB off the panel's colour about it: still all panel, no longer any
DEAD = 5                # a panel channel whose median is under this the type does not print (Darkness's red)
TABLE_STEP = 4          # levels a bin of faded_table
INK_CORE = 0.75         # ink share past which a pixel is a word's core
# Name bar. Its layout: a Stage card's ring and "Evolves from" strip take its left end; a Basic's bar runs on under
# the BASIC tab. Each is measured on five cards of far-apart types.
LAYOUT_REFS = {"staged": ["me02-013", "me01-060", "me02.5-043", "me02-056", "me01-003"],
               "basic": ["me01-086", "me02-004", "me02-041", "me02.5-116", "me03-055"]}
BAR_STRAY = 5           # RGB (median) a card's bar may stray from its type's other cards' and still be the type's
PROMO_STRAY = 12        # RGB (median) a promo's bar may stray from its type's and be on the Double Rare frame
RING = (70, 120, 46)                  # portrait inside the ring: centre x, y, radius; the card's own
STRIP = (100, 95, 480, 125)           # "Evolves from" strip: its words are the card's own
ICON_DISC = (669.2, 60.8, 28.0)       # name-bar icon and its rim
FIT_SX, FIT_SY = 18, 1.2              # px: the bar's colour is fitted this smooth along x and y
NAME_FILL = np.array([237, 232, 16], np.float32)   # the name's and ex logo's yellow, every type
FILL_NEAR = 50          # RGB distance from it a letter's fill pixel lies within
BAR_MARGIN, BAR_YELLOW_SHARE = 20, 0.02   # a bar with hardly any of it near the letters' yellow tells the fill by
                                          # its colour; else (Lightning, Dragon, Grass) by being enclosed
FIT_MIN = 0.05          # fit weight under which the bar's colour is read from a wider reach
ALPHA_RES = 12          # RGB rms off the slope's line within which the references' colours are explained by it
ALPHA_PRINT = 3         # px round a reference's print (name, HP, logo), where the slope says nothing
ALPHA_FILL = 2.0        # px: there the slope is taken from round it (Gaussian sigma)
EDGE_REACH = 2          # px from a frame piece within which a pixel's share of bar is its own (and the piece's
                        # colour is read)
EDGE_APART = 40         # RGB distance between bar and piece under which that share cannot be told
ROW_ICON_REACH = 3      # px past a row icon's rim that are the icon's (anti-aliasing, its soft shadow)
# Icon on every rarity: the icon's centre (measured to half a pixel), its disc, where its rim's colour and the
# rarity's surround are read along each radius; on the Mega Hyper Rare's etch, the symbol's part inside its etched ring
# and the icon's luminance that is all symbol (etched) / none.
ICON_C = (669.6, 60.8)
DISC_R = 26.3
RIM_AT, ROUND_AT = 25.0, 27.8
SYMBOL_R = 21.5
SYMBOL_LUM = (50, 90)
# White plate: white lettering on Darkness prints cut out of it at about this value (21..62 measured); a re-inked
# word's cut-out (plate under CUT_CLOSE) closes when the word turns dark, its soft edge CUT_EDGE px too.
CUT, CUT_CLOSE, CUT_EDGE = 35.0, 100, 3
WORDS = "retyped_words"     # layers kind: the words recolour re-inked (as retype.WORDS), for the plates
_build = threading.RLock()  # asset builders run one at a time (the GUI serves on threads; see _locked)

# Scarlet & Violet ex: up to ten Double Rares a type, spread over the sets (not Tera), and for the bar's layouts five
# Stage 2 and five Basic ones of far-apart types; the Stage 2 ones all of sv03.5, as the sets render the Stage tab a
# little apart (18..25 levels off one another; within a set under 1) and the typed mask would take that for the type.
# The icon (669.5, 60.5, r 26.6 measured on 35 prints), name and HP boxes, ring and strip are the Mega ex frame's.
SV_SOURCES = {
    "Grass": ["sv01-019", "sv02-027", "sv03-022", "sv04.5-005", "sv05-022", "sv06-023", "sv07-014", "sv08.5-011",
              "sv10-003", "sv10-025"],
    "Fire": ["sv02-037", "sv02-040", "sv03-033", "sv03.5-006", "sv03.5-038", "sv04-027", "sv05-034", "sv05-038",
             "sv08-037", "sv09-024"],
    "Water": ["sv02-052", "sv03.5-009", "sv04-050", "sv06-061", "sv07-030", "sv08-042", "sv09-043", "sv10-066",
              "sv10.5w-030", "me03-022"],
    "Lightning": ["sv01-065", "sv02-063", "sv03-073", "sv03.5-145", "sv04-070", "sv06-077", "sv08.5-028",
                  "sv08.5-032", "sv09-053", "sv10.5b-034"],
    "Psychic": ["sv01-086", "sv03-082", "sv03.5-065", "sv04-076", "sv04.5-029", "sv06-094", "sv08-076", "sv09-069",
                "sv10-081", "sv10.5w-045"],
    "Fighting": ["sv01-123", "sv02-117", "sv03-120", "sv03-124", "sv04-108", "sv05-099", "sv07-082", "sv08.5-051",
                 "sv09-079", "sv10-104"],
    "Darkness": ["sv01-131", "sv03-134", "sv03.5-024", "sv04.5-059", "sv06.5-036", "sv06.5-038", "sv08.5-064",
                 "sv09-098", "sv10-122", "sv10-139"],
    "Metal": ["sv01-143", "sv02-150", "sv03-153", "sv03-156", "sv04-135", "sv04-139", "sv04.5-066", "sv05-111",
              "sv07-105", "sv07-110"],
    "Dragon": ["sv02-153", "sv04-140", "sv04.5-069", "sv05-120", "sv05-122", "sv05-123", "sv09-114"],
    "Colorless": ["sv01-158", "sv03-164", "sv03.5-115", "sv04-156", "sv06-134", "sv08-147", "sv08.5-076",
                  "sv08.5-091", "sv10-150", "me02.5-162"],
}
SV_LAYOUT_REFS = {"staged": ["sv03.5-065", "sv03.5-003", "sv03.5-006", "sv03.5-009", "sv03.5-076"],
                  "basic": ["sv01-143", "me02-018", "me02.5-048", "sv01-123", "me02.5-162"]}
SV_RARITIES = {"Double rare": "dr", "Ultra Rare": "full", "Special illustration rare": "full", "Hyper rare": "full",
               "Shiny Ultra Rare": "full"}
SV_PANEL = (44, 518, 703, 873)       # the panel (the type's colour over the faded art), down to the weakness row
# Below it, round the weakness row, the rule box and the footer to the card's foot, and in the strips beside it, the
# band: the type's colour, a little lighter or darker from place to place (Lightning (255, 216, 49) throughout, Water and
# Darkness darker towards the foot), with grain, and the card's own art reaching over it; no foil. It is not the same
# pixels on each of the type's cards (the sets render it up to some 20 levels apart, and the art runs over it
# anywhere), so a type change moves what the old type's colour explains by the change of colour there (band_field).
SV_GLITTER = ((24, 873, 709, 990), (24, 518, 44, 873), (703, 518, 709, 873))
SV_FOOT_BOXES = (38, 962, 121, 993)  # the footer's regulation mark and set boxes (a white one on Darkness too)
OWN_GROW = 2                         # px round what keeps its colours on an SV ex (icons, badge): its soft rim
BAND_PRINT = 40                      # RGB off the band round it (median, 21 px) a pixel of print stands
# The frame's pieces on the band (the weakness row's rules, the rule box and its pill, the footer's boxes) are the same
# on every type: where the types' medians agree within BAND_AGREE (band_fixed). The type's colour is the median of its
# cards where they show band, BAND_SMOOTH px smooth (wider where none does); its spread, the band's own grain, the
# BAND_SPREAD percentile of their pixels' distance from it: a pixel within it is band, past twice it not.
BAND_AGREE, BAND_SMOOTH, BAND_SPREAD, PIECE_EDGE = 30, 6.0, 60, 3
# The footer's print (illustrator, boxes, number, rarity) is dark with a white keyline on every type, Darkness too. On a
# light band its dark ink stands out, the keyline within FOOT_KEYLINE px of it (on Metal's and Colorless's bands it is
# hardly lighter than the band). On Darkness's dark band the letters are the band's colour: they are told by their
# keyline, light (every channel over FOOT_WHITE, over FOOT_BOX_WHITE in the boxes, whose set box has a light grey inner
# line; grey within FOOT_GREY; FOOT_LIFT lighter than the band) beside dark (luminance under FOOT_DARK), with what it
# encloses (FOOT_HOLE px).
FOOT_WHITE, FOOT_BOX_WHITE, FOOT_GREY, FOOT_LIFT, FOOT_DARK, FOOT_HOLE, FOOT_KEYLINE = 200, 140, 40, 40, 90, 1500, 2
# A dark band (its colour's median luminance under BAND_DARK: Darkness's) shows the art under it in darker shades: a
# pixel darker than that (less than its spread lighter) is band whatever its hue. (The colour itself reads lighter where
# most of the type's cards print art over the band.)
BAND_DARK = 80
TERA_BOX = (200, 124, 640, 136)      # x0, y0, x1, y1: inside a Tera ex's rule box (silver), under the name bar
TERA_LIGHT, TERA_GREY = 0.6, 0.6     # shares of its pixels light (mean > 205) and grey (saturation < 40) on a Tera ex
TERA_WORDS = 0.03                    # ... and dark (mean < 100): its rule's words (a plain white background has none)


@dataclass(frozen=True, eq=False)
class Family:
    """A frame measured here: the Mega ex (MEGA) or the Scarlet & Violet ex (SV) it was drawn from."""
    key: str                  # its assets' folder
    sources: dict             # type -> its Double Rares (and promos on their frame)
    layout_refs: dict         # "staged" / "basic" -> Double Rares of far-apart types (bar_alpha, typed_mask)
    rarities: dict            # TCGdex rarity -> frame: "dr", "full" (the icon alone), "gold" (the icon and its etch)
    panel: tuple              # x0, y0, x1, y1
    glitter: tuple            # the boxes of the frame round the panel: Mega ex's glitter, SV ex's band
    yellow_name: bool         # name and ex logo yellow on every type (Mega ex); else dark, white on Darkness
    keep: tuple = ()          # boxes whose print is no word to re-ink (SV ex: the footer's boxes)
    band: bool = False        # the frame round the panel is a band of the type's colour over the art (SV ex, _band),
                              # not a glitter texture the same on each of the type's cards (Mega ex, _glitter)


MEGA = Family("mega", SOURCES, LAYOUT_REFS, FAMILY, PANEL, tuple(GLITTER), True)
SV = Family("sv", SV_SOURCES, SV_LAYOUT_REFS, SV_RARITIES, SV_PANEL, SV_GLITTER, False, keep=(SV_FOOT_BOXES,),
            band=True)


def family_of(card):
    """The frame family of an ex print. card: modern.read_card's dict of it."""
    return MEGA if card["name"].startswith("Mega ") else SV


def zone(boxes):
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def mode(a):
    """The most common value of a, rounded."""
    v, c = np.unique(np.round(a).astype(int), return_counts=True)
    return v[np.argmax(c)]


def _asset(name, fam=MEGA):
    return cache.path("assets", f"{fam.key}-v{ASSET_VERSION}", name)


def _locked(build):
    """An asset builder that runs one call at a time: the GUI serves requests on threads, and two first calls would
    both write its file while the other reads it (they call one another: the lock is re-entrant)."""
    @functools.wraps(build)
    def run(*args, **kwargs):
        with _build:
            return build(*args, **kwargs)
    return run


def ink(type_name):
    return compose.WHITE.copy() if type_name == "Darkness" else np.array(frame.DARK_INK, np.float32)


# ---------------------------------------------------------------- the print and its plates
def _card_space(img):
    img = img.convert("L").resize((foil.TEX, foil.TEX), Image.BICUBIC)
    return np.asarray(img, np.float32)[:, tcg.BLEED_X:tcg.BLEED_X + W].copy()


def etch_of(card_id):
    """The card's etch (_e) in card space, float32 0..255 (luminance)."""
    record = foil.printings(card_id)["base"]
    tex = foil.texture(record, record.get("_e"))
    if tex is None:
        return np.full((H, W), 255, np.float32)
    return _card_space(Image.fromarray(tex))


@functools.lru_cache(maxsize=24)
def layers(card_id):
    """(RGB, white plate, etch) of a card as printed, float32 card space."""
    return (tcg.scan(card_id)[..., :3].astype(np.float32), foil.plate(foil.printings(card_id)["base"]),
            etch_of(card_id))


def panel_weight(c, w, e, words=False, ramp=6, fam=MEGA, own=None):
    """How much of each pixel is panel: the white plate's flat panel value. words: with the words the plate cuts out
    (Darkness prints them white, cut out like its cost icons, Ability badge and red Ability names): white print cut
    out of the plate where nothing keeps its colours (own: default the etch's own marks, icons' rings and the badge's
    pill; an SV ex print has no etch, own_mask), with its soft edge. ramp: plate levels off the panel's value over
    which the weight falls to 0 (the panel's own grain and the soft overlays on it, light streaks, sit a few to some
    tens of levels off it)."""
    x0, y0, x1, y1 = PANEL_SAMPLE
    pv = mode(w[y0:y1, x0:x1])
    panel = zone([fam.panel])
    weight = np.clip(1 - (np.abs(w - pv) - 2) / ramp, 0, 1) * panel
    if own is None:
        own = dilate(e < 200, 3)                                    # icons' rings, the badge's pill
    if words:
        red = c[..., 0] - c[..., 1:].max(-1) > 60
        white = (c.min(-1) > 150) & (c.max(-1) - c.min(-1) < 60)
        letters = white & (w < WORD_CUT) & ~own & panel & ~zone(fam.keep)
        weight = np.maximum(weight, dilate(letters, 2) & ~own & ~dilate(red, 1) & panel & ~zone(fam.keep))
    return weight


def own_mask(c, card, fam):
    """What keeps its colours on an SV ex print, which has no etch to mark it: the printed cost icons (found on it,
    inplace.cost_icons), the weakness row's icons and the Ability badge's red pill with its word, OWN_GROW px round
    them. bool HxW"""
    from . import inplace
    out = row_icons(card) > 0
    r = inplace.cost_r(card)
    yy, xx = np.mgrid[:H, :W]
    kinds = {}
    for a in card.get("attacks") or []:
        for t in a.get("cost") or []:
            kinds[t] = kinds.get(t, 0) + 1
    for kind, count in kinds.items():
        for cx, cy in inplace.cost_icons(c, card, kind, count):
            out |= np.hypot(xx - cx, yy - cy) <= r + 1
    red = (c[..., 0] - c[..., 1:].max(-1) > 60) & zone([fam.panel])
    out |= fill_holes(dilate(red, 1))
    return dilate(out, OWN_GROW)


def glitter_weight(w, fam=MEGA):
    return np.clip((140 - w) / 60, 0, 1) * zone(fam.glitter)


# ---------------------------------------------------------------- per type, from its cards
def _colour_lut(colours):
    """Distance (RGB levels) from each LUT bin to the nearest bin holding the colours."""
    n = 256 // LUT_STEP
    idx = np.clip(colours // LUT_STEP, 0, n - 1).astype(int)
    count = np.zeros((n, n, n), np.int32)
    np.add.at(count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    dist = np.where(count >= 3, 0.0, 1e3).astype(np.float32)
    for _ in range(16):
        pad = np.pad(dist, 1, constant_values=1e3)
        best = dist
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx or dy or dz:
                        step = np.sqrt(dx * dx + dy * dy + dz * dz)
                        best = np.minimum(best, pad[1 + dz:1 + dz + n, 1 + dy:1 + dy + n, 1 + dx:1 + dx + n] + step)
        dist = best
    return dist * LUT_STEP


def lut_dist(lut, c):
    n = lut.shape[0]
    idx = np.clip(c // LUT_STEP, 0, n - 1).astype(int)
    return lut[idx[..., 0], idx[..., 1], idx[..., 2]]


def _quantiles(glit):
    """(luminance at each of QUANT quantiles, the colours there) of glitter colours (N x 3)."""
    lum = luminance(glit)
    order = np.argsort(lum)
    idx = ((np.arange(QUANT) + 0.5) / QUANT * (len(order) - 1)).astype(int)
    half = max(1, len(order) // (QUANT * 2))
    ramp = np.stack([np.median(glit[order[max(0, i - half):i + half + 1]], 0) for i in idx])
    return lum[order[idx]], ramp


def _glitter_px(c, w, fam=MEGA):
    """The glitter's own colours on a print: plate dark, no print's or splash's edge (N x 3)."""
    med = cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 21).astype(np.float32)
    g = (glitter_weight(w, fam) > 0.99) & (np.abs(c - med).max(-1) < 40) & ~zone([FOOT_PRINT, ROW_TEXT])
    return c[g]


@functools.lru_cache(maxsize=None)
@_locked
def params(type_name, fam=MEGA):
    """(panel tint (3,), glitter luminance quantiles (QUANT,), glitter colours there (QUANT x 3), glitter LUT); the
    glitter's three are None on a family without glitter (SV: its band, band_field)."""
    from . import fairy
    out = _asset(f"params-{type_name}{fairy.tag(type_name)}.npz", fam)
    if not out.exists():
        glitter = fam.glitter and not fam.band
        if type_name in DERIVED:
            tint = fairy.colour(params(DERIVED[type_name], fam)[0][None])[0]
            if glitter:
                texture, have = glitter_template(type_name, fam)
                glit = texture[have]
                lum, ramp = _quantiles(glit)
                np.savez(out, tint=tint, lum=lum, ramp=ramp, lut=_colour_lut(glit))
            else:
                np.savez(out, tint=tint)
        else:
            tops, mids, glit = [], [], []
            for card_id in fam.sources[type_name]:
                c, w, e = layers(card_id)
                med = cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 31).astype(np.float32)
                m = (panel_weight(c, w, e, fam=fam) > 0.99) & (np.abs(c - med).max(-1) < 45) & (e > 200)
                tops.append(np.percentile(c[m], 99.5, axis=0))      # multiply: the art's white shows the tint itself
                mids.append(np.median(c[m], axis=0))
                if glitter:
                    glit.append(_glitter_px(c, w, fam))
            tops = np.sort(np.array(tops), 0)
            tint = tops[-2] if len(tops) >= 3 else tops[-1]       # art without white under the panel reads low
            tint[np.median(mids, 0) < DEAD] = 0         # Darkness's red: none whatever the art (the top is splashes)
            if glitter:
                glit = np.concatenate(glit)
                lum, ramp = _quantiles(glit)
                np.savez(out, tint=tint, lum=lum, ramp=ramp, lut=_colour_lut(glit))
            else:
                np.savez(out, tint=tint)
    d = np.load(out)
    if "lum" not in d:
        return d["tint"], None, None, None
    return d["tint"], d["lum"], d["ramp"], d["lut"]


@functools.lru_cache(maxsize=None)
@_locked
def faded_table(k, fam=MEGA):
    """The faded art's channel k as a table of its other two (TABLE_STEP levels a bin, filled and smoothed): from the
    panels of every type that prints all three (panel / tint). A type that prints none of k (Darkness: red) takes
    its art's k from it."""
    out = _asset(f"faded-{k}.npy", fam)
    if out.exists():
        return np.load(out)
    rest = [i for i in range(3) if i != k]
    n = 256 // TABLE_STEP
    total, count = np.zeros((n, n)), np.zeros((n, n))
    for type_name, ids in fam.sources.items():
        tint = params(type_name, fam)[0]
        if (tint < 1).any():
            continue
        for card_id in ids:
            c, w, e = layers(card_id)
            med = cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 9).astype(np.float32)
            plain = (panel_weight(c, w, e, fam=fam) > 0.99) & (np.abs(c - med).max(-1) < 30)      # not words, icons
            f = np.clip(c[plain] / tint * 255, 0, 255)
            idx = np.clip(f[:, rest] // TABLE_STEP, 0, n - 1).astype(int)
            np.add.at(total, (idx[:, 0], idx[:, 1]), f[:, k])
            np.add.at(count, (idx[:, 0], idx[:, 1]), 1)
    table, have = None, None
    for sigma in (1.5, 4, 12):                  # bins with little to go on read from wider round them
        num, den = cv2.GaussianBlur(total, (0, 0), sigma), cv2.GaussianBlur(count, (0, 0), sigma)
        est = num / np.maximum(den, 1e-6)
        table = est if table is None else np.where(have < 0.5, est, table)
        have = den if have is None else np.maximum(have, den)
    table = table.astype(np.float32)
    np.save(out, table)
    return table


def faded_channel(f, k, fam=MEGA):
    """Channel k of the faded art f (HxWx3, the other two read) from faded_table, bilinear."""
    rest = [i for i in range(3) if i != k]
    tab = faded_table(k, fam)
    n = tab.shape[0]
    u = np.clip(f[..., rest[0]] / TABLE_STEP - 0.5, 0, n - 1)
    v = np.clip(f[..., rest[1]] / TABLE_STEP - 0.5, 0, n - 1)
    u0, v0 = np.minimum(u.astype(int), n - 2), np.minimum(v.astype(int), n - 2)
    du, dv = u - u0, v - v0
    return ((1 - du) * (1 - dv) * tab[u0, v0] + du * (1 - dv) * tab[u0 + 1, v0]
            + (1 - du) * dv * tab[u0, v0 + 1] + du * dv * tab[u0 + 1, v0 + 1])


def ramp_at(ramp, q):
    return np.stack([np.interp(q, np.arange(QUANT), ramp[:, k]) for k in range(3)], -1)


@functools.lru_cache(maxsize=None)
@_locked
def glitter_template(type_name, fam=MEGA):
    """The type's glitter texture: per pixel, the median of its cards that show glitter there (white plate dark,
    colour among the type's glitter colours). -> (HxWx3, bool HxW where any card shows it)"""
    from . import fairy
    out = _asset(f"glitter-{type_name}{fairy.tag(type_name)}.png", fam)
    if not out.exists():
        if type_name in DERIVED:
            texture, have = glitter_template(DERIVED[type_name], fam)
            texture = np.where(have[..., None], fairy.colour(texture), 0)
        else:
            lut = params(type_name, fam)[3]
            stack = []
            for card_id in fam.sources[type_name]:
                c, w, e = layers(card_id)
                ok = (w < GLIT_PLATE) & zone(fam.glitter) & (lut_dist(lut, c) <= GLIT_NEAR)
                ok = cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # no print's edge in it
                stack.append(np.where(ok[..., None], c, np.nan))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # pixels no card shows: nan, not had
                texture = np.nanmedian(np.stack(stack), 0)
            have = ~np.isnan(texture[..., 0])
            texture = np.nan_to_num(texture)
        save_rgba(np.dstack([texture, have * 255.0]), out)
    rgba = load_rgba(out)
    return rgba[..., :3], rgba[..., 3] > 127


@functools.lru_cache(maxsize=None)
@_locked
def band_fixed(fam=SV):
    """bool HxW: the frame's pieces on an SV ex's band, the same on every type (the weakness row's rules, the rule box
    and its pill, the footer's boxes): where the medians of each type's cards agree within BAND_AGREE, and their dark
    outline, dark on every type whose band is light (PIECE_EDGE px round them; on a dark band it is the band's
    colour, and the cards' prints sit a fraction of a pixel apart there)."""
    out = _asset("band-fixed.png", fam)
    if not out.exists():
        band = zone(fam.glitter)
        meds = np.stack([np.median(np.stack([layers(c)[0] for c in ids]), 0) for ids in fam.sources.values()])
        m = ((meds.max(0) - meds.min(0)).max(-1) < BAND_AGREE) & band
        lums = luminance(meds)
        light = [l for l in lums if np.median(l[band & ~m]) >= BAND_DARK]
        m |= (np.max(light, 0) < FOOT_DARK) & dilate(m, PIECE_EDGE) & band
        Image.fromarray(m.astype(np.uint8) * 255).save(out)
    return np.asarray(Image.open(out)) > 127


@functools.lru_cache(maxsize=None)
@_locked
def band_field(type_name, fam=SV):
    """(the type's colour over an SV ex's band, HxWx3: per pixel the median of the type's cards where they show band,
    not its pieces, print or row icons, BAND_SMOOTH px smooth; its spread, the band's own grain: how far those
    pixels lie from it, BAND_SPREAD percentile). A derived type's (Fairy) is its base type's in its colours."""
    from . import fairy, modern
    out = _asset(f"band-{type_name}{fairy.tag(type_name)}.npz", fam)
    if not out.exists():
        band = zone(fam.glitter)
        if type_name in DERIVED:
            base, spread = band_field(DERIVED[type_name], fam)
            colour = np.where(band[..., None], fairy.colour(base), 0)
        else:
            fixed = dilate(band_fixed(fam), 2)
            cs, shown = [], []
            for card_id in fam.sources[type_name]:
                c = layers(card_id)[0]
                cs.append(c)
                shown.append(band & ~fixed & ~dilate(_band_print(c), 2)
                             & (row_icons(modern.read_card(card_id), reach=4) == 0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # pixels no card shows: nan
                med = np.nanmedian(np.stack([np.where(m[..., None], c, np.nan) for c, m in zip(cs, shown)]), 0)
            have = ~np.isnan(med[..., 0])
            steps = [(BAND_SMOOTH * k, BAND_SMOOTH * k) for k in (1, 4, 16)]
            colour = _fit([np.nan_to_num(med)], [have], steps) * band[..., None]
            spread = float(np.percentile(np.concatenate([np.abs(c[m] - colour[m]).max(-1) for c, m in zip(cs, shown)]),
                                         BAND_SPREAD))
        np.savez_compressed(out, colour=np.round(np.clip(colour, 0, 255)).astype(np.uint8), spread=spread)
    d = np.load(out)
    return d["colour"].astype(np.float32), float(d["spread"])


# ---------------------------------------------------------------- name bar
def layout(card):
    return "basic" if not card.get("stage") else "staged"


def bar_zone(lay="staged", fam=MEGA):
    """The name bar's rows, less (on a Stage card) the portrait, the "Evolves from" strip and the glitter sliver left
    of the ring."""
    if lay == "basic":
        return zone([BAR])
    yy, xx = np.mgrid[:H, :W]
    cx, cy, r = RING
    return zone([BAR]) & ~(np.hypot(xx - cx, yy - cy) < r) & ~zone([STRIP]) & ~zone(fam.glitter)


@functools.lru_cache(maxsize=None)
@_locked
def typed_mask(lay="staged", fam=MEGA):
    """Name-bar pixels that take the type's colour: where the layout's cards of different types differ."""
    out = _asset(f"typed-{lay}.png", fam)
    if not out.exists():
        cards = [layers(c)[0] for c in fam.layout_refs[lay]]
        votes = sum((np.abs(cards[0] - o).max(-1) > 30).astype(int) for o in cards[1:])
        m = (votes >= 3) & bar_zone(lay, fam)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        m = drop_small(m, 200)                 # specks on the silver pieces (tab lettering) are not the bar
        Image.fromarray(m.astype(np.uint8) * 255).save(out)
    return np.asarray(Image.open(out)) > 127


def _icon_cover(r):
    yy, xx = np.mgrid[:H, :W]
    cx, cy, _ = ICON_DISC
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0, 1).astype(np.float32)


def print_cover(card, kinds=("name", "hp", "logo")):
    """Where the composers set the card's name, HP and ex logo (they match these prints to a pixel or two). float HxW"""
    from . import modern                        # modern edits with this module
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # a long name squeezed to fit
        return modern._cover(modern._drawn(modern._name, card), kinds)


@functools.lru_cache(maxsize=64)
def _source_cover(card_id, kinds=("name", "hp", "logo")):
    from . import modern
    return print_cover(modern.read_card(card_id), kinds)


def hp_print(cover_hp):
    """bool HxW: the HP and its label as printed, give or take the composers' pixel or two."""
    from . import modern
    return dilate(cover_hp > 0.05, modern.REACH - 2)


@functools.lru_cache(maxsize=None)
@_locked
def bar_alpha(lay="staged", fam=MEGA):
    """How much of each name-bar pixel is the bar, the rest the frame's silver pieces over it (their edges are
    anti-aliased): the slope of the layout's references' colours on their own bar fits. They share the pieces, not the
    bar, so the slope is 1 on the bar and 0 on a piece. Where a reference prints (name, HP, logo), or the slope does
    not explain the colours, the typed mask stands. float HxW"""
    from . import modern
    out = _asset(f"alpha-{lay}.npy", fam)
    if out.exists():
        return np.load(out)
    ids = fam.layout_refs[lay]
    cs = np.stack([layers(c)[0] for c in ids])
    gs = np.stack([_fit_bar([layers(c)[0]], [modern.read_card(c)], tcg.card(c)["types"][0], solid=False, fam=fam)
                   for c in ids])
    e, d = cs - cs.mean(0), gs - gs.mean(0)
    a = (e * d).sum((0, 3)) / np.maximum((d * d).sum((0, 3)), 1)
    res = np.sqrt(((e - a[None, ..., None] * d) ** 2).sum((0, 3)) / len(ids))
    printed = dilate(np.max([_source_cover(c) for c in ids], 0) > 0.05, ALPHA_PRINT)
    held = ~printed & (res < ALPHA_RES) & bar_zone(lay, fam)
    a = np.where(held, np.clip(a, 0, 1), 0).astype(np.float32)
    num = cv2.GaussianBlur(a, (0, 0), ALPHA_FILL)          # under the print: the slope round it
    den = cv2.GaussianBlur(held.astype(np.float32), (0, 0), ALPHA_FILL)
    near = np.where(den > 0.3, num / np.maximum(den, 1e-6), typed_mask(lay, fam))
    alpha = (np.where(held, a, near) * bar_zone(lay, fam)).astype(np.float32)
    np.save(out, alpha)
    return alpha


def card_alpha(c, g, lay="staged", fam=MEGA):
    """bar_alpha, with the card's own share of bar along the frame's pieces' edges: the cards' prints sit a fraction
    of a pixel apart, so an edge pixel's share is solved from its colour between the bar's (g, a fit) and the piece's
    beside it (c - s = a (g - s)), where the two differ enough to tell."""
    a = bar_alpha(lay, fam)
    piece = a < 0.05
    edge = dilate(piece, EDGE_REACH) & ~piece & bar_zone(lay, fam)
    w = piece.astype(np.float32)
    s = cv2.GaussianBlur(c * w[..., None], (0, 0), EDGE_REACH) / np.maximum(
        cv2.GaussianBlur(w, (0, 0), EDGE_REACH), 1e-6)[..., None]
    gs = g - s
    den = (gs * gs).sum(-1)
    own = np.clip(((c - s) * gs).sum(-1) / np.maximum(den, 1), 0, 1)
    use = edge & (den > EDGE_APART ** 2)
    return np.where(use, own, a).astype(np.float32)


def bar_core(c, g, type_name, card, fam=MEGA):
    """The print's solid pixels in the name and HP boxes: the letters' dark outline and yellow fill (Mega ex), or the
    letters, dark (white on Darkness), and the whole ex logo (SV ex); the HP's ink. g: the bar's colour there (a fit).
    Everything else is bar, its grain too, save the rim round these. Only where the composers set the name, logo and
    HP: not the frame's pieces' dark edges, or their shadow, beside it."""
    from . import modern
    lum = luminance(c)
    lmed = luminance(g)                         # the bar's own (a median round white HP rides up past a dark bar)
    hp = hp_print(print_cover(card, ("hp",)))
    boxes = ((zone([NAME_BOX]) & dilate(print_cover(card, ("name", "logo")) > 0.05, modern.REACH))
             | (zone([HP_BOX]) & hp))
    dark = boxes & (lum < np.minimum(100, np.maximum(lmed - 45, 0.6 * lmed)))
    if not fam.yellow_name and type_name == "Darkness":
        dark[:] = False         # an SV ex prints its name and HP white on Darkness: dark there is the bar's own shade
    if fam.yellow_name:
        near_fill = (np.linalg.norm(g[boxes & typed_mask(layout(card), fam)] - NAME_FILL, axis=-1)
                     < FILL_NEAR + BAR_MARGIN)
        if near_fill.mean() < BAR_YELLOW_SHARE:
            fill = boxes & (np.linalg.norm(c - NAME_FILL, axis=-1) < FILL_NEAR)
        else:
            fill = fill_holes(dark, max_area=150) & ~dark
        core = dark | fill
    else:
        core = dark | (zone([NAME_BOX]) & (print_cover(card, ("logo",)) > 0.3))
        if type_name == "Darkness":
            core |= zone([NAME_BOX]) & (lum > lmed + 60) & name_print(card)
    if type_name == "Darkness":
        core |= zone([HP_BOX]) & (lum > lmed + 60) & hp
    return core


def name_print(card):
    """bool HxW: the name as printed, give or take the composers' pixel or two (SV ex: dark or white like the HP)."""
    from . import modern
    return dilate(print_cover(card, ("name",)) > 0.05, modern.REACH - 2)


def _fit(cs, frees, steps=((FIT_SX, FIT_SY), (4 * FIT_SX, 4 * FIT_SY), (16 * FIT_SX, 16 * FIT_SY))):
    """The colour of the prints cs where frees (bool HxW each) say, smooth: Gaussian sigmas steps (x, y), each wider
    one only where the closer has (almost) nothing to go on."""
    out, weak = None, None
    for sx, sy in steps:
        num, den = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
        for c, free in zip(cs, frees):
            free = free.astype(np.float32)
            num += cv2.GaussianBlur(c * free[..., None], (0, 0), sx, sigmaY=sy)
            den += cv2.GaussianBlur(free, (0, 0), sx, sigmaY=sy)
        g = num / np.maximum(den, 1e-6)[..., None]
        if out is None:
            out, weak = g, den < FIT_MIN * len(cs)
        else:                                   # where the closer fit has (almost) nothing to go on
            out = np.where(weak[..., None], g, out)
            weak &= den < FIT_MIN * len(cs)
    return out


def _fit_bar(cs, cards, type_name, solid=True, fam=MEGA):
    """The bar colour, smooth, from the print-free bar pixels of the prints cs (cards: their data): a first fit
    leaving out the print's box generously, then one leaving out the print's solid pixels only. solid: only where the
    bar is all bar (bar_alpha), not on the frame pieces' soft edges (bar_alpha's own references take the typed mask)."""
    from . import modern
    icon = _icon_cover(ICON_DISC[2] + 3) > 0
    typed = [(bar_alpha(layout(card), fam) > 0.95 if solid else typed_mask(layout(card), fam)) & ~icon
             for card in cards]
    rough = [m & ~dilate(zone([NAME_BOX, HP_BOX]) & (luminance(c) < 100), 4)
             & ~dilate(print_cover(card) > 0.05, modern.REACH + 2) for m, c, card in zip(typed, cs, cards)]
    g = _fit(cs, rough)
    return _fit(cs, [m & ~dilate(bar_core(c, g, type_name, card, fam), 2) for m, c, card in zip(typed, cs, cards)])


def _source_bar(card_id, type_name, fam=MEGA):
    from . import modern
    return _fit_bar([layers(card_id)[0]], [modern.read_card(card_id)], type_name, fam=fam)


def _common(fam=MEGA):
    """Bar pixels every layout has all bar, away from the print and the icon: where two cards' bars are compared."""
    return ((bar_alpha("staged", fam) > 0.95) & (bar_alpha("basic", fam) > 0.95) & ~zone([NAME_BOX, HP_BOX])
            & (_icon_cover(ICON_DISC[2] + 3) == 0))


def _stray(a, b, fam=MEGA):
    common = _common(fam)
    return float(np.median(np.abs(a[common] - b[common]).max(-1)))


@functools.lru_cache(maxsize=None)
@_locked
def type_bar(type_name, fam=MEGA):
    """The type's bar colour (rows 0..BAR), fitted on its cards whose bar is the type's: each card's own fit near at
    least half of the others' (me04-015's and me01's Basics' stray)."""
    from . import modern
    from . import fairy
    out = _asset(f"bar-{type_name}{fairy.tag(type_name)}.npy", fam)
    if not out.exists():
        if type_name in DERIVED:
            bar = fairy.colour(type_bar(DERIVED[type_name], fam))
        else:
            ids = fam.sources[type_name]
            fits = {c: _source_bar(c, type_name, fam) for c in ids}
            near = lambda a, b: _stray(fits[a], fits[b], fam) <= BAR_STRAY
            kept = [c for c in ids if len(ids) < 3 or 2 * sum(near(c, o) for o in ids if o != c) >= len(ids) - 1]
            bar = _fit_bar([layers(c)[0] for c in kept], [modern.read_card(c) for c in kept], type_name, fam=fam)
        np.save(out, bar[:BAR[3]].astype(np.float32))
    bar = np.zeros((H, W, 3), np.float32)
    bar[:BAR[3]] = np.load(out)
    return bar


_frames = {}


def tera(card_id):
    """Whether the print is a Tera ex: its Tera rule set in a silver box under the name bar (light grey, dark words)."""
    c = tcg.scan(card_id)[..., :3]
    x0, y0, x1, y1 = TERA_BOX
    box = np.clip(c[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    sat = cv2.cvtColor(box, cv2.COLOR_RGB2HSV)[..., 1]
    lum = box.mean(-1)
    return bool(TERA_LIGHT < (lum > 205).mean() < 0.95 and (sat < 40).mean() > TERA_GREY
                and (lum < 100).mean() > TERA_WORDS)


def frame_of(card_id, card):
    """The print's frame: "dr", "full" (Ultra Rare, Special Illustration Rare; an SV ex's Hyper Rare and Shiny Ultra
    Rare too), "gold" (Mega Hyper Rare), or None (a frame not measured: a promo of its own design, a Black White Rare,
    a Tera ex). card: modern.read_card's dict of it as printed."""
    if card_id not in _frames:
        fam = family_of(card)
        rarity = tcg.card(card_id).get("rarity")
        if fam is SV and tera(card_id):
            frame = None
        elif rarity in fam.rarities:
            frame = fam.rarities[rarity]
        elif any(card_id in ids for ids in fam.sources.values()):
            frame = "dr"
        elif rarity == "Promo" and card["type"] in fam.sources:   # the print's bar against the type's, off its print
            frame = "dr" if _stray(layers(card_id)[0], type_bar(card["type"], fam), fam) <= PROMO_STRAY else None
        else:
            frame = None
        _frames[card_id] = frame
    return _frames[card_id]


def supports(card_id, card):
    """Whether a type change can be made on this print: a Mega ex or a Scarlet & Violet ex on a frame measured here,
    with the TCG Live extract (the plates tell panel, glitter and what lies over them apart). card: modern.read_card's
    dict of it."""
    if not tcg.TCGLIVE or card.get("family") != "ex":
        return False
    try:
        return frame_of(card_id, card) is not None
    except ValueError:                          # no TCG Live printing
        return False


def recolour_bar(c, card, old_t, new_t, words):
    fam = family_of(card)
    lay = layout(card)
    typed = typed_mask(lay, fam)
    g_old = _fit_bar([c], [card], old_t, fam=fam)
    g_new = type_bar(new_t, fam)
    shift = card_alpha(c, g_old, lay, fam)[..., None] * (g_new - g_old)
    core = bar_core(c, g_old, old_t, card, fam)
    near = dilate(core, 2) & ~core & typed
    _, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core)
    alpha = core.astype(np.float32)
    if len(ys):
        at = lab[near] - 1
        pc = c[ys[at], xs[at]]
        num = np.linalg.norm(c[near] - g_old[near], axis=-1)
        den = np.maximum(np.linalg.norm(pc - g_old[near], axis=-1), 30)
        alpha[near] = np.clip(num / den, 0, 1)
    out = c + (1 - alpha)[..., None] * shift
    # the print's soft shadow on the bar (round Darkness's white HP): a shade of the bar takes that shade of the new
    shaded = np.zeros((H, W), bool)
    if len(ys):
        cn, gn = c[near], g_old[near]
        k = np.clip((cn * gn).sum(-1) / np.maximum((gn * gn).sum(-1), 1), 0, 1)
        dark = (np.abs(cn - k[:, None] * gn).max(-1) < MIX_OFF) & (k > SHADE_MIN) & (k < 0.95)
        ny, nx = np.nonzero(near)
        shaded[ny[dark], nx[dark]] = True
        out[shaded] = c[shaded] + k[dark, None] * (g_new - g_old)[shaded]
    # HP and its label (and an SV ex's name): dark ink on every type but Darkness (white)
    if (old_t == "Darkness") != (new_t == "Darkness"):
        inked = zone([HP_BOX]) & dilate(core, 2) & hp_print(print_cover(card, ("hp",)))
        if not fam.yellow_name:
            logo = dilate(print_cover(card, ("logo",)) > 0.05, 1)
            inked |= zone([NAME_BOX]) & dilate(core, 2) & name_print(card) & ~logo
        i_old, i_new = ink(old_t), ink(new_t)
        to = i_old - g_old
        share = np.clip(((c - g_old) * to).sum(-1) / np.maximum((to * to).sum(-1), 1), 0, 1)
        share = np.where(inked & ~shaded, share, 0)[..., None]
        out = np.where(share > 0, share * i_new + (1 - share) * (c + shift), out)
        np.maximum(words, share[..., 0], out=words)
    return swap_icon(out, new_t, edge=False, fam=fam)


# ---------------------------------------------------------------- icon (every rarity)
def _radial(img, r_to):
    """img sampled along each pixel's radius from ICON_C, at radius r_to (bilinear)."""
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    cx, cy = ICON_C
    r = np.maximum(np.hypot(xx - cx, yy - cy), 1e-3)
    mx, my = cx + (xx - cx) * r_to / r, cy + (yy - cy) * r_to / r
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR)


@functools.lru_cache(maxsize=None)
def icon_donor(type_name, fam=MEGA):
    """A Double Rare print of the type, for its name-bar icon (the Mega Evolution regular cards' artwork) and the bar
    round it: its first source; a derived type's (Fairy) is its base type's in its colours, with its own icon."""
    if type_name in DERIVED:
        from . import fairy
        canvas = np.dstack([fairy.colour(layers(fam.sources[DERIVED[type_name]][0])[0]), np.full((H, W), 255.0)])
        cx, cy, _ = meframe.ICON
        put_sprite(canvas, meframe.icon(type_name, "name")[0], cx, cy, snap=True)
        return canvas[..., :3].astype(np.float32)
    return layers(fam.sources[type_name][0])[0]


def swap_icon(c, new_t, edge=True, fam=MEGA):
    """The name-bar icon on the new type: its disc from a Double Rare of the type; edge: its anti-aliased rim blended
    again with the card's own surround (an Ultra Rare's white ring, a Mega Hyper Rare's gold); else the donor's rim
    (a Double Rare's bar)."""
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(xx - ICON_C[0], yy - ICON_C[1])
    donor = icon_donor(new_t, fam)
    if not edge:
        a = _icon_cover(ICON_DISC[2])[..., None]
        return a * donor + (1 - a) * c
    a = np.clip(DISC_R + 0.5 - r, 0, 1)[..., None]
    rim, round_ = _radial(donor, RIM_AT), _radial(c, ROUND_AT)
    disc = np.where((a > 0) & (a < 1), rim, donor)
    return np.where(a > 0, a * disc + (1 - a) * round_, c)


def symbol_etch(new_t):
    """Mega Hyper Rare etch inside its ring: the new type's symbol, etched (255) where the icon's symbol is dark."""
    lum = luminance(icon_donor(new_t))
    lo, hi = SYMBOL_LUM
    return np.clip((hi - lum) / (hi - lo), 0, 1) * 255


def etch(texture, card, new_t):
    """The card's etch texture (TCG Live's 1024x1024 layout, bleed and all) for the new type: a Mega Hyper Rare's
    symbol inside its etched ring."""
    if texture is None or texture.shape[0] != foil.TEX or frame_of(card["id"], card) != "gold":
        return texture
    out = texture.copy()
    view = out[:, tcg.BLEED_X:tcg.BLEED_X + W]
    yy, xx = np.mgrid[:H, :W]
    a = np.clip(SYMBOL_R + 0.5 - np.hypot(xx - ICON_C[0], yy - ICON_C[1]), 0, 1)
    sym = symbol_etch(new_t)
    for ch in range(min(3, view.shape[2])) if view.ndim == 3 else [None]:
        plane = view if ch is None else view[..., ch]
        plane[:] = (a * sym + (1 - a) * plane).astype(plane.dtype)
    return out


# ---------------------------------------------------------------- the card
def row_icons(card, reach=ROW_ICON_REACH):
    """Coverage (HxW) of the weakness row's energy icons where the composers set them on SV / ME ex prints
    (compose.SV_ROW_ICONS): the card's weakness, resistance and retreat icons from its data, reach px past the rim."""
    discs = [compose.SV_ROW_ICONS[side][:3] for side in ("weakness", "resistance") if card.get(side)]
    x0, y, r, pitch = compose.SV_ROW_ICONS["retreat"]
    discs += [(x0 + i * pitch, y, r) for i in range(int(card.get("retreat") or 0))]
    yy, xx = np.mgrid[:H, :W]
    out = np.zeros((H, W), np.float32)
    for cx, cy, r in discs:
        out = np.maximum(out, np.clip(r + reach + 0.5 - np.hypot(xx - cx, yy - cy), 0, 1))
    return out


def edge_share(x, p, q, w, w_q, pv, fallback):
    """Share of panel of edge pixels x (Nx3) between the panel (colour p, plate pv) and a cut-out beside it (colour
    q, plate w_q): the panel's own colour darker (an icon's soft shadow on it) takes that shade; a blend of the two
    colours the blend's share; else the plate's share between the two values; else fallback."""
    span = pv - w_q
    out = np.where(np.abs(span) > EDGE_SPAN, np.clip((w - w_q) / np.where(span == 0, 1, span), 0, 1), fallback)
    d = p - q
    dd = (d * d).sum(-1)
    mix = np.clip(((x - q) * d).sum(-1) / np.maximum(dd, 1), 0, 1)
    off_mix = np.abs(x - (q + mix[:, None] * d)).max(-1)
    out = np.where((dd > EDGE_APART ** 2) & (off_mix < MIX_OFF), mix, out)
    k = np.clip((x * p).sum(-1) / np.maximum((p * p).sum(-1), 1), 0, 1)
    off_shade = np.abs(x - k[:, None] * p).max(-1)
    return np.where((off_shade < MIX_OFF) & (off_shade < off_mix) & (k > SHADE_MIN), k, out)


def retint(c, s, i_old, t_old, t_new, fam=MEGA):
    """The panel under words of ink i_old, share s (HxWx1), on the new type's tint: the faded art the old tint was
    multiplied over, times the new one. A channel the old type does not print (Darkness's red) is read from the
    other two (faded_table)."""
    dead = t_old < 1
    if not dead.any():
        return (c - s * i_old) * t_new / t_old
    under = (c - s * i_old) / np.maximum(1 - s, 0.05)
    f = np.clip(under / np.maximum(t_old, 1) * 255, 0, 255)
    for k in np.flatnonzero(dead):
        f[..., k] = faded_channel(f, k, fam)
    return (1 - s) * f * t_new / 255


def _panel(c, w, e, old_t, new_t, words, fam=MEGA, own=None):
    """The text panel on the new type's tint, its words re-inked. own: what keeps its colours (panel_weight). -> (RGB,
    the words' ink share before)"""
    out = c.copy()
    t_old, t_new = params(old_t, fam)[0], params(new_t, fam)[0]
    i_old, i_new = ink(old_t), ink(new_t)
    lum = luminance(c)
    wp = panel_weight(c, w, e, words=(i_old > 128).all(), ramp=PLATE_RAMP, fam=fam, own=own)[..., None]
    panel = zone([fam.panel])
    kept = zone(fam.keep) | (own if own is not None else False)
    med = cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 25).astype(np.float32)
    lbg, link = luminance(med), luminance(i_old[None])[0]
    s_raw = np.clip((lbg - lum) / np.maximum(lbg - link, 1) if old_t != "Darkness"
                    else (lum - lbg) / np.maximum(link - lbg, 1), 0, 1)
    core = (s_raw > INK_CORE) & (wp[..., 0] > 0.5) & ~kept
    s = np.where(dilate(core, 2) & ~kept, s_raw, 0)[..., None]
    solid = wp[..., 0] > 0.99
    out = np.where(solid[..., None], s * i_new + retint(c, s, i_old, t_old, t_new, fam), out)
    # the panel's soft edge round what keeps its colours (icons, badge, splashes): each pixel takes its share of
    # the panel's change (edge_share)
    ring = dilate(solid, RING_PX) & ~solid & panel
    part = ~solid & panel
    plain = (solid & (s[..., 0] == 0)).astype(np.float32)
    est, have = None, None
    for sigma in (EDGE_PX, 4 * EDGE_PX, 16 * EDGE_PX):          # the panel's colour about each pixel
        num = cv2.GaussianBlur(c * plain[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(plain, (0, 0), sigma)
        e_s = num / np.maximum(den, 1e-6)[..., None]
        est = e_s if est is None else np.where((have < 0.2)[..., None], e_s, est)
        have = den if have is None else np.maximum(have, den)
    change = retint(est, np.zeros_like(s), i_old, t_old, t_new, fam) - est
    x0, y0, x1, y1 = PANEL_SAMPLE
    pv = mode(w[y0:y1, x0:x1])
    far = ~dilate(solid, RING_PX)
    _, lab = cv2.distanceTransformWithLabels(far.astype(np.uint8) ^ 1, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    fy, fx = np.nonzero(far)
    # the panel's grain and soft overlays (glows): its share by the plate's ramp, or by its colour: a pixel the
    # panel's own colour is panel whatever the plate says
    looks = np.clip(1 - (np.abs(c - est).max(-1) - LOOKS_NEAR) / LOOKS_RAMP, 0, 1) * panel
    share = np.maximum(wp[..., 0], looks * (s[..., 0] == 0))
    ry, rx = np.nonzero(ring)
    if len(fy) and len(ry):
        at = lab[ry, rx] - 1
        share[ry, rx] = edge_share(c[ry, rx], est[ry, rx], c[fy[at], fx[at]], w[ry, rx], w[fy[at], fx[at]], pv,
                                   share[ry, rx])
    py, px = np.nonzero(part | ring)
    out[py, px] = c[py, px] + share[py, px, None] * change[py, px]
    np.maximum(words, (s[..., 0] * solid).astype(np.float32), out=words)
    return out, s_raw


def _band_print(c):
    """Print on an SV ex's band, whatever its ink: what stands out from the band round it (words, figures, their
    keylines). bool HxW"""
    med = cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 21).astype(np.float32)
    return np.abs(c - med).max(-1) > BAND_PRINT


def _glitter(c, out, w, card, old_t, new_t, fam=MEGA):
    """Pixels in the old type's glitter colours take the new type's texture (or its colour at the same quantile); the
    weakness row's icons keep theirs (a Colorless icon's light disc can be in a type's glitter colours)."""
    _, lum_old, ramp_old, lut_old = params(old_t, fam)
    _, _, ramp_new, _ = params(new_t, fam)
    q = np.interp(luminance(c), lum_old, np.arange(QUANT))
    fit_old, mapped = ramp_at(ramp_old, q), ramp_at(ramp_new, q)
    t_old, have_old = glitter_template(old_t, fam)
    t_new, have_new = glitter_template(new_t, fam)
    by_colour = np.clip(1 - (lut_dist(lut_old, c) - GLIT_NEAR) / GLIT_RAMP, 0, 1)
    base_old = np.where(have_old[..., None], t_old, fit_old)
    wsh = glitter_weight(w, fam) * by_colour
    target = np.where(have_new[..., None], t_new + (c - base_old) * 0.5, mapped)
    wsh *= 1 - row_icons(card, reach=0)
    return wsh[..., None] * target + (1 - wsh[..., None]) * out


def _band(c, out, card, old_t, new_t, fam=SV):
    """An SV ex's band on the new type: a pixel its old type's colour explains (within the band's spread; less and less
    so to twice it), or on a dark band any darker shade, moves by the change of colour there, its grain scaled to the
    new type's (never grown: a dark type's mottling would show as specks on a light one). The frame's pieces, the
    row's icons, the footer's print and art unlike the band keep theirs."""
    t_old, spread_old = band_field(old_t, fam)
    t_new, spread_new = band_field(new_t, fam)
    d = np.abs(c - t_old).max(-1)
    band = zone(fam.glitter) & ~band_fixed(fam)
    lum_t = float(np.median(luminance(t_old)[band]))
    shade = (luminance(c) < lum_t + spread_old) if lum_t < BAND_DARK else np.zeros(d.shape, bool)   # a dark band's shades
    share = (np.maximum(np.clip(2 - d / spread_old, 0, 1), shade) * band
             * ~_foot_print(c, t_old, (d > spread_old) & ~shade) * (1 - row_icons(card, reach=0)))[..., None]
    target = t_new + (c - t_old) * min(1.0, spread_new / spread_old)
    return share * target + (1 - share) * out


def _foot_print(c, band, unlike):
    """bool HxW: an SV ex's footer print (illustrator, regulation and set boxes, number, rarity), dark with a white
    keyline on every type: dark ink unlike the band (unlike: bool HxW; not a dark band's own shades) with its keyline
    round it, and a keyline lighter than the band (band: its colour) with what it encloses."""
    foot = zone([FOOT_PRINT, SV_FOOT_BOXES])
    dark = (luminance(c) < FOOT_DARK) & foot
    light = ((c.min(-1) > np.where(zone([SV_FOOT_BOXES]), FOOT_BOX_WHITE, FOOT_WHITE))
             & (c.max(-1) - c.min(-1) < FOOT_GREY) & (luminance(c) > luminance(band) + FOOT_LIFT))
    keyline = light & dilate(dark, 2) & foot
    return (fill_holes(keyline, max_area=FOOT_HOLE) | dilate(dark & unlike, FOOT_KEYLINE)) & foot


def _row_words(out, card, s_raw, old_t, new_t, fam=MEGA):
    """The weakness row's words and values ("weakness", "×2", "-30") in the new type's ink; the row's icons keep
    theirs. The footer keeps its print on every type (an SV ex's: dark with a white keyline, Darkness's too)."""
    icons = row_icons(card) > 0
    where = zone([ROW_TEXT]) & ~zone(fam.keep) & ~icons
    core = where & (s_raw > INK_CORE)
    s = np.where(dilate(core, 2) & where, s_raw, 0)[..., None]
    ground = cv2.inpaint(np.clip(out, 0, 255).astype(np.uint8), dilate(core, 3).astype(np.uint8), 3,
                         cv2.INPAINT_TELEA).astype(np.float32)
    return np.where(s > 0, s * ink(new_t) + (1 - s) * ground, out)


def recolour(rgba, card, new_type, plate=None):
    """The card (RGBA float32 733x1024) on the new type's frame. card: modern.read_card's dict of it as printed (its
    type, Stage, name, HP, weakness row); plate: the print's white plate (default: its own). The words re-inked are
    recorded under the layers kind WORDS (foil.build_modern cuts them out of the plate, or closes them)."""
    card_id, old_t = card["id"], card["type"]
    fam = family_of(card)
    c = rgba[..., :3].astype(np.float32)
    words = np.zeros((H, W), np.float32)
    out = rgba.copy()
    if frame_of(card_id, card) != "dr":            # no type colour on the frame: the icon only
        out[..., :3] = swap_icon(c, new_type, fam=fam)
    else:
        w = plate if plate is not None else foil.plate(foil.printings(card_id)["base"])
        e = etch_of(card_id)
        own = own_mask(c, card, fam) if not fam.yellow_name else None    # an SV ex: no etch marks its icons
        rgb, s_raw = _panel(c, w, e, old_t, new_type, words, fam, own)
        rgb = _band(c, rgb, card, old_t, new_type, fam) if fam.band else _glitter(c, rgb, w, card, old_t, new_type, fam)
        if (old_t == "Darkness") != (new_type == "Darkness"):
            rgb = _row_words(rgb, card, s_raw, old_t, new_type, fam)
        out[..., :3] = recolour_bar(rgb, card, old_t, new_type, words)
    from . import layers
    with layers.kind(WORDS):
        layers.record(words)
    return out


def plate(img, old_t, new_t, words, fam=MEGA):
    """The white plate of a card made new_t: white lettering is cut out of it on Darkness; dark the foil runs
    through, so a Darkness card's cut-out words that turn dark are closed (the panel's value under them)."""
    words = np.clip(words, 0, 1)
    if new_t == "Darkness" and old_t != "Darkness":
        return img * (1 - words) + CUT * words
    if old_t == "Darkness" and new_t != "Darkness":
        cut = (words > 0.3) & (img < CUT_CLOSE)
        if cut.any():
            cut = dilate(cut, CUT_EDGE)            # the cut's soft edge too: closed from the plate past it
            closed = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), cut.astype(np.uint8), 3,
                                 cv2.INPAINT_TELEA).astype(np.float32)
            x0, y0, x1, y1 = PANEL_SAMPLE
            closed = np.where(zone([fam.panel]), mode(img[y0:y1, x0:x1]), closed)   # dark words: the panel's
            return np.where(cut, closed, img)
    return img


def build_all():
    """Build and cache every type's assets, of both families (minutes, once; needs the TCG Live extract)."""
    from . import fairy
    for fam in (MEGA, SV):
        for lay in fam.layout_refs:
            bar_alpha(lay, fam)
        for t in TYPES:
            if t in DERIVED and not fairy.available():
                continue
            params(t, fam)
            band_field(t, fam) if fam.band else glitter_template(t, fam)
            type_bar(t, fam)
        for k in range(3):
            faded_table(k, fam)
