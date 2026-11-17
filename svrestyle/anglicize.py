"""A Japanese Mega ex print set in English (local.py's Japanese cards): its words are taken off and set again with
the composers, in the places and inks the English Mega ex prints have them, so that the result is the card "as
printed" for modern.py: every later edit finds its text where the composers set it.

Japanese and English Mega ex prints share the frame to a pixel (the Japanese scans, resized to 733x1024, sit within
half a pixel of me01-086's weakness row and rule box), but not the words' places:
- name bar: the Japanese name ends at a fixed place before its ex logo; the English name is set by the composers and
  the logo follows it. The logo is lifted off the print (frame.ex_logo's shape, found where the print's dark "ex"
  strokes lie) and set after the English name; the bar under the old name and logo is rebuilt by LaMa. The HP is the
  same in both and stays.
- Stage tab and "Evolves from" strip: the English tab is pasted (stage.draw), the Japanese line is taken off its
  strip (modern.fill_strip) and the English one set.
- Mega badge: one line in Japanese, two in English (taller): the rebuilt English badge (megarule) is set over it.
- abilities and attacks: everything standing out in the text zone is taken off (modern.moves_ink) and LaMa rebuilds
  the art and panel; the English block is set in the Absol ex layout, in the old text's ink.
- weakness row: Japanese prints set the labels, icons and values in other places (the resistance icon 5 px left, the
  weakness icon 7 px right): their slots between the row's rules are taken off whole (all but the Retreat circles)
  and the English row set with its labels. LaMa rebuilds the slots: these bands carry the art (mej7-067's yellow to
  pink streaks), which a blend along the rows (modern.fill_band) leaves in blocks; no label is left for LaMa to copy.
- rule box: the English box (me01-086's, every Mega ex prints the same) is laid over the Japanese one.
- footer: kept as printed (the illustrator line is already English; set code M7 and the number are the Japanese
  set's: the English ones are not known yet).
"""
import warnings

import cv2
import numpy as np

from . import cache, compose, frame, inpaint, layers, local, megarule, modern, stage, tcg
from .imaging import dilate, fill_holes, load_rgba, luminance, save_rgba

W, H = 733, 1024
VERSION = 1                  # bump when the English print changes, so cached ones are made again
NAME_ZONE = (128, 26, 500, 102)      # x0, y0, x1, y1: the Japanese name and ex logo (the HP starts right of it)
LOGO_SEARCH = (300, 20, 520, 110)    # where the logo's dark strokes are looked for
LOGO_DARK = 75               # luminance under which a pixel is the logo's dark fill (or a name glyph's outline)
LOGO_CORE = 5                # px the logo shape is eroded to its dark strokes, and the strokes opened to drop outlines
LOGO_MATCH = 0.4             # least match of the strokes for the logo to be found (the prints: 0.51-0.55; a dark
                             # bar, me01-086's, where the strokes cannot be told from it: 0.29)
LOGO_GROW = 2                # px round the logo shape cut with it (as modern.card_logo cuts it)
NAME_GROW = 3                # px round the old name's glyphs taken off with them (outline, anti-aliasing)
# The Japanese weakness row's print (labels; icons and values), as x0, y0, x1, y1 boxes (mej7-030, mej7-067): its
# frame between them (the bar after the weakness, the swoosh before the retreat) is left alone.
# Between the row's two rules (y 877-880, 911-914): boxes reaching them would ring the print and fill whole.
JA_ROW = ((40, 881, 100, 911), (104, 881, 200, 911), (222, 881, 300, 911), (300, 881, 398, 911), (448, 881, 516, 911))
RULE_ZONE = (240, 918, 722, 1006)    # x0, y0, x1, y1 round the rule box
RULE_REF = "me01-086"        # Mega Absol ex: an English rule box laid over the Japanese one
RULE_AGREE = 26              # RGB levels within which two prints show the same box
RULE_SMOOTH = 9              # px: the box's shape is opened with a disc this wide (its edge, where bands differ)
RULE_FEATHER = 1.0           # px blur of the box's edge as it is laid on


def _logo_shape():
    (_, alpha, _, _, _), _, _ = frame.ex_logo()
    return (alpha > 0.5).astype(np.uint8)


def find_logo(scan):
    """(x, y): where frame.ex_logo's shape lies on the print's ex logo (its top-left corner), by its dark strokes."""
    shape = _logo_shape()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * LOGO_CORE + 1,) * 2)
    strokes = cv2.erode(shape, kernel).astype(np.float32)
    x0, y0, x1, y1 = LOGO_SEARCH
    dark = (luminance(scan[y0:y1, x0:x1, :3]) < LOGO_DARK).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    match = cv2.matchTemplate(dark.astype(np.float32), strokes, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(match)
    if not best >= LOGO_MATCH:                    # NaN: nothing dark at all
        raise ValueError(f"no ex logo found on the name bar (best match {best:.2f})")
    return x0 + x, y0 + y


def logo_sprite(scan, at):
    """The print's ex logo as compose.name_bar_text takes it ((crop, alpha, x, y, origin), left x, right x), and its
    shape on the print (bool HxW). Placed as frame.ex_logo places Absol's: by its left edge after the name."""
    shape = _logo_shape()
    x, y = at
    h, w = shape.shape
    where = np.zeros((H, W), bool)
    where[y:y + h, x:x + w] = shape > 0
    grown = dilate(where, LOGO_GROW)
    ys, xs = np.nonzero(grown)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = cv2.GaussianBlur(grown[y0:y1, x0:x1].astype(np.float32), (3, 3), 0.7)
    cols = np.flatnonzero(where.any(axis=0))
    sprite = (scan[y0:y1, x0:x1, :3].copy(), alpha, x0, y0, (x0, y0))
    return (sprite, int(cols[0]), int(cols[-1]) + 1), grown


def name_off(scan, logo):
    """bool HxW: the Japanese name and the ex logo (what stands out on the bar left of the HP)."""
    ink = modern.ink_in(scan, NAME_ZONE)
    return dilate(ink, NAME_GROW) | dilate(logo, 1)


def row_off():
    """bool HxW: the Japanese weakness row's labels, icons and values (not the Retreat circles): their slots whole.
    The band behind them is busy (art runs under it, mej7-067's yellow-to-pink): what stands out there is not only
    the print, and the slots hold nothing else."""
    out = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in JA_ROW:
        out[y0:y1, x0:x1] = True
    return out


def evolves_off(scan):
    """bool HxW: the Japanese "Evolves from" line on its strip."""
    x0, y0, x1, y1 = stage.STRIP
    ink = modern.ink_in(scan, (x0, y0 - 4, min(x1, megarule.BADGE_LEFT), y1 + 4), holes=False)
    return dilate(ink, 1) & stage.strip_shape(scan)


def rule_box():
    """(RGB, alpha) over RULE_ZONE: RULE_REF's rule box (English), cached. Its shape is where it agrees with at least
    two of local.py's prints (the box and its frame are the same on every Mega ex, its words only on the English
    ones; the band round it is each card's own), closed, filled and smoothed."""
    path = cache.path("local", f"v{VERSION}", "rule-box.png")
    x0, y0, x1, y1 = RULE_ZONE
    if path.exists():
        img = load_rgba(path)
        return img[..., :3], img[..., 3] / 255
    ref = tcg.scan(RULE_REF)[y0:y1, x0:x1, :3]
    votes = sum((np.abs(ref - local.raw_scan(cid)[y0:y1, x0:x1, :3]).max(axis=-1) < RULE_AGREE).astype(np.uint8)
                for cid in local.CARDS)
    agree = cv2.morphologyEx((votes >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(agree)
    if n < 2:
        raise ValueError("the English rule box was not found")
    box = fill_holes(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    box = cv2.morphologyEx(box, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RULE_SMOOTH,) * 2))
    box = cv2.erode(box, np.ones((3, 3), np.uint8)) > 0          # its rim is half band
    alpha = cv2.GaussianBlur(box.astype(np.float32), (0, 0), RULE_FEATHER)
    save_rgba(np.dstack([ref, alpha * 255]), path)
    return ref, alpha


def put_rule_box(canvas):
    x0, y0, x1, y1 = RULE_ZONE
    rgb, alpha = rule_box()
    a = alpha[..., None]
    canvas[y0:y1, x0:x1, :3] = canvas[y0:y1, x0:x1, :3] * (1 - a) + rgb * a


def put_badge(canvas, card):
    x0, y0, x1, y1 = megarule.BADGE
    rgb, alpha = megarule.rebuilt_badge(card)
    a = alpha[..., None]
    canvas[y0:y1, x0:x1, :3] = canvas[y0:y1, x0:x1, :3] * (1 - a) + rgb * a


def _ink(scan, box, where=None):
    found = modern.print_ink(scan, box, where)
    return found if found is not None else (np.array(frame.DARK_INK, np.float32), None)


def english(scan, card):
    """The Japanese print (float32 RGBA 733x1024) set in English for card (modern.read_card's dict). -> (RGBA,
    {kind: coverage} of what was set)."""
    logo, logo_mask = logo_sprite(scan, find_logo(scan))
    moves = modern.moves_ink(scan, card["family"])
    moves_fill, moves_edge = _ink(scan, modern.MOVES_ZONE[card["family"]], moves)
    row_fill, row_edge = _ink(scan, modern.ROW_LABELS)
    canvas = inpaint.inpaint(scan, dilate(name_off(scan, logo_mask) | moves, modern.MARK_GROW) | row_off())
    if card["stage"]:
        canvas = modern.fill_strip(canvas, evolves_off(scan))
    put_rule_box(canvas)
    with layers.recording() as drawn:
        stage.draw(canvas, card, ("tab",), printed=bool(card["stage"]))
        if card["stage"]:
            modern._evolves(canvas, card)
        put_badge(canvas, card)
        modern._name(canvas, card, logo, mega=True)
        shrink = modern._moves(canvas, card, moves_fill, moves_edge)
        compose.weakness_row(canvas, card, row_fill, row_edge, labels=True, icons=compose.SV_ROW_ICONS,
                             retreat=(0, 0))
    if shrink is not None and shrink < 1:
        warnings.warn(f"{card['id']}: ability/attack text set at {shrink:.0%} to fit")
    canvas[..., 3] = scan[..., 3]
    return canvas, drawn


def english_print(card_id):
    """local.py's Japanese card set in English: float32 RGBA 733x1024, cached."""
    path = cache.path("local", f"v{local.VERSION}", f"{card_id}-english-v{VERSION}.png")
    if path.exists():
        return load_rgba(path)
    card = modern.read_card(card_id)
    out, _ = english(local.raw_scan(card_id), card)
    save_rgba(out, path)
    return out
