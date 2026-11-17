"""Holofoil masks made for local.py's cards, which Pokemon TCG Live does not print yet: the white plate and etch of a
Mega ex Double Rare (SvUltra, foil.ME_ULTRA), drawn from the card's own print the way the game's Mega ex plates
are laid out, so foil.build_modern edits them as it edits the game's (local.printings gives them as a printing).

White plate (0 no foil .. 255 all foil), card space:
- foil over the whole face: art, frame, name bar (the name and ex logo lit at the bar's value, as on Absol ex's
  plate), the Mega badge (a printed badge keeps its foil) and the rule box;
- cut out (black), as foil.EX_CUT cuts them on ex plates: the HP and the name-bar type icon; the abilities and
  attacks (words, cost icons, Ability badge and names, damage), found where the print has them
  (modern.moves_as_printed); the weakness row's labels, values and icons; the footer (illustrator, regulation mark,
  set box, number, rarity, copyright); the rule box's words; and the Stage pieces' print (the tab's lettering, the
  "Evolves from" line, the portrait).
Etch (TCG Live's 1024x1024 layout, white = plain, black = etched; foil.ETCHED is the line between):
- the art between the name bar and the weakness row is etched along its own lines: its edges (luminance gradient)
  etched the deeper the sharper, so the foil's texture follows the drawing, as an etched card's does;
- words, icons and badges are never etched (cut out of the art's etch, foil.etch_words);
- the HP is etched black on a plain box, as the Mega ex etches print it (foil.etch_reprinted re-etches an edited HP).
Everything else (frame, row, rule box, footer) is plain.
"""
import functools

import cv2
import numpy as np
from PIL import Image

from . import cache, compose, local, megarule, meframe, modern, stage, tcg
from .imaging import dilate, drop_small, fill_holes, largest_filled, luminance

W, H, TEX = 733, 1024, 1024
VERSION = 1                   # bump when the plates change, so cached ones are made again
CUT_GROW = 1                  # px round cut-out print (its anti-aliased rim)
CUT_SOFT = 0.7                # px blur of the cut-outs' edges
SNAP = 2                      # px the print may lie off where the composers set it (stand-in for modern.REACH)
ICON_DISC = (669.2, 60.8, 28.5)          # the name bar's type icon and its rim (megaretype.ICON_DISC)
HP_BOX = (492, 28, 643, 100)             # x0, y0, x1, y1 round the HP and its label
FOOTER = (38, 934, 250, 996)             # illustrator, regulation mark, set box, number, rarity (left of the rule box)
COPYRIGHT = (200, 999, 540, 1016)
FOOTER_DIFF = 80              # how far footer print stands out from the art round it
COPYRIGHT_DIFF = 40           # and the copyright line, darker than it
RULE_TEXT = (262, 922, 712, 1000)        # inside the rule box
RULE_CLOSE = 9                # px: the yellow field and the black pill closed over their letters
RULE_DARK = 100               # luminance under which a pixel of the field is a letter, of the pill the pill
SPECK = 6                     # px: smaller standouts are the scan's glitter, not print
ART = (102, 868)                         # y range of the art the etch etches (under the bar, above the row)
WORD_CLEAR = 4                # px round print kept clear of the art's etch (the print's own edges are no art)
ETCH_SIGMA = 1.2              # px the print is smoothed over before its edges are read
ETCH_FULL = 0.85              # edge strength (share of the art's 98th percentile) etched all the way
ETCH_FLOOR = 0.12             # weaker edges are not etched at all
ETCH_DEPTH = 225              # etch value taken off at full strength (255 -> 30)
ETCH_SOFT = 0.6               # px blur of the etched lines
PLAIN = 255


def _print(card_id):
    return tcg.scan(card_id), modern.read_card(card_id)


def _snap(scan, cover):
    """The print near what the composers set (cover: coverage 0..1), with what they set: bool HxW."""
    if not (cover > 0.05).any():
        return np.zeros((H, W), bool)
    return modern.ink_near(scan, cover, reach=SNAP) | (cover > 0.3)


def standout(scan, box, diff=None, darker=False):
    """bool HxW: print in box that stands out from what is round it by more than FOOTER_DIFF (stronger than
    modern.INK_DIFF: the art under the footer is busy, and a Japanese scan's glitter specks stand out too), its
    specks dropped. Strokes only: holes would fill the art's rings. darker: only print darker than round it (the
    copyright line's grey)."""
    x0, y0, x1, y1 = box
    rgb = np.clip(scan[..., :3], 0, 255).astype(np.uint8)
    pad = modern.MEDIAN
    crop = cv2.copyMakeBorder(rgb[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_REFLECT)
    lift = crop.astype(np.int16) - cv2.medianBlur(crop, modern.MEDIAN).astype(np.int16)
    off = -lift.min(axis=2) if darker else np.abs(lift).max(axis=2)
    ink = drop_small((off > (diff or FOOTER_DIFF))[pad:-pad, pad:-pad], SPECK)
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = ink
    return out


def rule_words(scan):
    """bool HxW: the rule box's words: dark letters on its yellow field, yellow letters on its black pill (the box,
    its rim and the pill stay foil)."""
    x0, y0, x1, y1 = RULE_TEXT
    rgb = scan[y0:y1, x0:x1, :3]
    lum = luminance(rgb)
    yellow = (rgb[..., 0] > 150) & (rgb[..., 1] > 140) & (rgb[..., 2] < 120)
    close = lambda m, k: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    if not yellow.any():
        return np.zeros((H, W), bool)
    field = largest_filled(close(yellow, RULE_CLOSE))          # not the pill's yellow letters
    dark_letters = field & (lum < RULE_DARK)
    pill = fill_holes(close(lum < RULE_DARK, RULE_CLOSE)) & ~field
    words = dark_letters | (pill & yellow)
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = drop_small(words, SPECK)
    return out


def _disc(cx, cy, r):
    yy, xx = np.mgrid[:H, :W]
    return np.hypot(xx - cx, yy - cy) <= r


@functools.lru_cache(maxsize=8)
def marks(card_id):
    """{kind: bool HxW}: what the print cuts out of the foil, by kind (foil.EX_CUT's kinds)."""
    scan, card = _print(card_id)
    out = {}
    name = modern._drawn(modern._name, card)
    out["hp"] = _snap(scan, name.get("hp", np.zeros((H, W), np.float32)))
    box = np.zeros((H, W), bool)
    x0, y0, x1, y1 = HP_BOX
    box[y0:y1, x0:x1] = True
    out["hp"] &= box
    out["type_icon"] = _disc(*ICON_DISC)
    for kind, cover in modern.moves_as_printed(scan, card).items():
        out[kind] = _snap(scan, cover)
    row = modern._drawn(lambda c, k: compose.weakness_row(c, k, compose.WHITE, None, labels=True,
                                                          icons=compose.SV_ROW_ICONS), card)
    out["row_icon"] = dilate(row.get("row_icon", np.zeros((H, W), np.float32)) > 0.1, 1)
    out["row_text"] = _snap(scan, row.get("row_text", np.zeros((H, W), np.float32)))
    out["footer"] = standout(scan, FOOTER) | standout(scan, COPYRIGHT, COPYRIGHT_DIFF, darker=True)
    for x0, y0, x1, y1, _ in compose.footer_box_rects(True, False):     # the regulation mark's box, whole
        out["footer"][int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    out["rule_text"] = rule_words(scan)
    if card["stage"]:
        out["evolves"] = _snap(scan, modern._drawn(modern._evolves, card).get("evolves",
                                                                             np.zeros((H, W), np.float32)))
        out["portrait"] = _disc(*meframe.PORTRAIT_DISC)
    out["stage_text"] = stage._lettering(scan[..., :3]) & stage._letter_box()
    return out


def white_plate(card_id):
    """The base printing's white plate in card space (float32 0..255), cached."""
    path = cache.path("local", f"v{local.VERSION}", f"{card_id}-wp-v{VERSION}.png")
    if path.exists():
        return np.asarray(Image.open(path), np.float32)
    scan, _ = _print(card_id)
    cut = np.zeros((H, W), bool)
    for m in marks(card_id).values():
        cut |= m
    cover = cv2.GaussianBlur(dilate(cut, CUT_GROW).astype(np.float32), (0, 0), CUT_SOFT)
    out = PLAIN * (1 - np.clip(cover, 0, 1)) * (scan[..., 3] / 255)
    Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8)).save(path)
    return out.astype(np.float32)


def art_etch(scan):
    """Etch values (card space, 0..255) of the art etched along its own lines."""
    lum = cv2.GaussianBlur(scan[..., :3].astype(np.float32).mean(axis=-1), (0, 0), ETCH_SIGMA)
    edge = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1))
    y0, y1 = ART
    full = float(np.percentile(edge[y0:y1], 98)) * ETCH_FULL or 1.0
    depth = np.clip((edge / full - ETCH_FLOOR) / (1 - ETCH_FLOOR), 0, 1)
    depth = cv2.GaussianBlur(depth, (0, 0), ETCH_SOFT)
    out = np.full((H, W), PLAIN, np.float32)
    out[y0:y1] = PLAIN - ETCH_DEPTH * depth[y0:y1]
    return out


def etch(card_id):
    """The base printing's etch, 1024x1024 L (TCG Live's layout: the card plus bleed), cached."""
    path = cache.path("local", f"v{local.VERSION}", f"{card_id}-etch-v{VERSION}.png")
    if path.exists():
        return np.asarray(Image.open(path))
    scan, card = _print(card_id)
    found = marks(card_id)
    out = art_etch(scan)
    words = np.zeros((H, W), bool)                 # never etched: cut out of the art's etch
    for kind, m in found.items():
        if kind != "hp":
            words |= m
    out[dilate(words, WORD_CLEAR)] = PLAIN
    x0, y0, x1, y1 = megarule.BADGE                # the badge: plain, as the printed foil keeps it
    out[y0:y1, x0:x1] = np.where(megarule.badge()[..., 3] > 25, PLAIN, out[y0:y1, x0:x1])
    hp_box = np.zeros((H, W), bool)
    hx0, hy0, hx1, hy1 = HP_BOX
    hp_box[hy0:hy1, hx0:hx1] = True
    out[hp_box] = PLAIN                            # the HP etched black on a plain box
    out[found["hp"]] = 0
    right = TEX - W - tcg.BLEED_X
    tex = np.pad(np.clip(out, 0, 255).round().astype(np.uint8), ((0, 0), (tcg.BLEED_X, right)), mode="edge")
    Image.fromarray(tex).save(path)
    return tex


def plate(record):
    return white_plate(record["_local"])


def field(record):
    """The plate's foil with nothing printed on it (foil.field): all foil over the face."""
    return PLAIN * tcg.scan(record["_local"])[..., 3] / 255


def texture(record, name):
    return etch(record["_local"]) if name == record.get("_e") else None
