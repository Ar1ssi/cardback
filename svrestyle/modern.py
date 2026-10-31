"""Edit Scarlet & Violet and Mega Evolution cards in place: the card keeps its own scan, frame and art; only what
was edited is taken off and set again in the same layouts the restyles use.

- Name and HP, and the weakness row (weakness, resistance, retreat): the original is set again with the composers
  (they match these prints to a pixel or two, see tests) to find exactly where it is printed.
- Stage, "Evolves from" and the evolution portrait: the Stage pieces sit at one place on every SV / ME card, so the
  ones that changed are pasted over (stage.py); back to Basic they are taken off and the BASIC tab set. A Stage card
  keeps its own strip (Mega ex cards end it under their badge): the old "Evolves from" line is found like the name and
  rebuilt from the strip's own shading, row by row and column by column (fill_strip).
- Abilities and attacks: ex prints do not stack their blocks by one rule, so the old block is found on the card
  instead: everything in the text zone that stands out from the panel behind it (text, cost icons, the Ability
  badge); on a textured panel (Ancient and Future prints), only what stands out round the old block (moves_off).
  The holofoil, which needs each kind of it apart, takes the composers' layout moved onto the print
  (moves_as_printed). Regular cards then take the new block in the Mega Evolution layout (Scarlet & Violet regular
  cards share it), ex cards in the Absol ex layout, in the old block's own ink (print_ink: its fill colour, and its
  keyline when the text is outlined: over full art, on Ancient and Future Pokemon).
- Type: a regular card's print is lifted off its type's frame and set on the new type's (retype.py), in the new type's
  ink; what is set again takes that ink too. A Mega ex is recoloured on its own print first (megaretype.py), and that
  print is the card as printed for every other edit.
- Abilities and attacks whose Energy types or damages alone change (costs, {X} in effect text, a damage) keep their
  print: the icons, symbols and figures are set again where they are printed (inplace.py).
- The Mega Evolution rule ("mega_rule", megarule.py) is set over the print, under the "Evolves from" strip; a Mega ex
  print keeps its own badge on top of it. Nothing is taken off for it.
What is taken off is rebuilt by LaMa (cached per print and set of edited parts), except in the weakness row, which
is filled along its pixel rows (fill_band), and on the "Evolves from" strip (fill_strip). Energy written {G}, {R} ...
in effect text prints as its symbol.
"""
import functools
import hashlib
import warnings

import cv2
import numpy as np

from . import (cache, compose, edit, frame, inpaint, inplace, layers, mecompose, megaretype, megarule, meframe, retype,
               stage, tcg)
from .imaging import dilate, fill_holes, load_rgba, luminance, save_rgba

W, H = 733, 1024
MODERN = ("sv", "me")
GROUPS = {"name": ("prefix", "name", "hp"), "moves": ("abilities", "attacks"),
          "row": ("weakness", "resistance", "retreat"), "stage": ("stage", "evolves_from", "portrait"),
          "type": ("type",), "rule": ("mega_rule",)}

# Abilities and attacks sit in this zone: x0, y0, x1, y1 (under the window bar / Pokedex strip, above the weakness
# row).
MOVES_ZONE = {"regular": (36, 512, 697, 868), "ex": (30, 524, 703, 870)}
MEDIAN = 45          # px: wider than a glyph or cost icon, so the median is the panel behind them
INK_DIFF = 38        # how far a pixel stands out from that panel to count as printed
INK_CLOSE = 5        # px: joins glyph strokes and icon rims into whole shapes
MARK_GROW = 3        # px around everything taken off (anti-aliasing, keylines)
REACH = 5            # px: name, HP and weakness-row print may sit this far from where the composers set them
ROW_NEAR = 4         # REACH for the weakness row
ROW_LABELS = (30, 878, 530, 914)   # x0, y0, x1, y1: the row's "weakness", "resistance", "retreat" (always printed)
WHOLE_REACH = 3      # x reach: a printed shape touching the reach is taken whole this far out
RULE = 40            # px: a straight run of print at least this long is a bar's rule, not a glyph
# LaMa sometimes invents a blot in a hole on a plain panel (sv04-128: a brown smudge where a Colorless energy was).
GHOST_PLAIN = 9.0      # luminance spread of the panel around a blot below which the panel is plain
GHOST_RING = (3, 9)    # px: that panel is read this far out from the blot
GHOST_MIN = 80         # px: smaller standouts are the panel's own grain
GHOST_MAX = 4000       # px: bigger standouts are not blots (art LaMa rebuilt)
GHOST_GROW = 5         # px around a blot covered too: its soft edge is not caught as standing out
CLEAN_VERSION = 24     # bump when cleaned() changes what it makes, so cached cleanups are made again
ICON_SHADOW = 3        # px: a printed row icon's shadow reaches this far out (and down) from it
# The composers stack a block of abilities and attacks by their own rule, which ex prints do not keep: where the old
# block is printed, its layout is moved onto the print, move by move and then line by line (moves_as_printed).
MOVE_SHIFT = 80      # px a move may lie above or below where the composers set it (me02.5-152's Ability: 53)
LINE_SHIFT = 3       # px a line of it may lie further (the print's line pitch is not quite the composers')
LINE_JOIN = 2        # px: rows of ink this close are one line
MOVE_GAP = 20        # px: a gap this tall between two lines starts another move (lines: 5-8, header to body: 16)
HEADERS = ("badge", "icon")   # kinds only the first line of a move holds
TEXT_REACH = 2       # px round the old text (moved onto the print) its letters may reach (old_text)
PRINT_NEAR = 6       # px round the old block (moved onto the print) its print and outline reach (moves_off)
LINE_RUN = 90        # px a printed line may run on past where the composers end it: a word (moves_off)
TEXTURED = 0.09      # share of the text zone standing out away from the old block past which the panel is textured
                     # (plain panels: its rim and specks, 0.02-0.04; Ancient and Future: 0.15-0.53)
# The old block's ink is read from its glyphs: shapes a glyph's size, lined up on a shared baseline.
GLYPH_H = (5, 30)    # px: tallest and shortest glyph (accented capitals, commas aside)
GLYPH_W = 40         # px: wider shapes are art or words run together
GLYPH_MIN = 8        # px: smaller shapes are grain
TEXT_LINE = 8        # glyphs whose bottoms sit within a pixel of each other make a line of text
KEYLINE_RING = 2     # px: an outline is read this far out from the glyphs
KEYLINE_MAX = 6      # px: and measured out to this far, ring by ring (full art's run 2.4 to 3.1 px, me02-109, sv03.5-199)
KEYLINE_NEAR = 60    # RGB off the outline's colour a ring pixel of it lies within
INK_SNAP = 48        # RGB distance within which a measured ink is the stock ink (white, DARK_INK)
BAND_GAP, BAND_SAMPLE = 3, 6   # px: weakness-row gaps take their colour this far out, from this many pixels
BAND_REACH = 24      # px past BAND_GAP those pixels are looked for
BAND_SMOOTH = 5      # rows a weakness-row gap's blend is the median over
BAND_STEP = 30       # RGB levels apart past which two rows' blends are not one stretch of band (a rule's edge)
BAND_SEARCH = 80     # px past the nearest place a weakness-row gap looks for clean band to take its grain from
BAND_PIECE = 16      # px: gaps take their grain in square pieces this size, each from its own nearest clean place
PRINT_GROW = 2       # px: print, icons whole, grown by its soft edge, is kept out of that grain
PRINT_DIFF = 70      # how far print stands out from the weakness row; its grain stays under (sv10.5b-046: 52)
# The old "Evolves from" line is taken off its strip STRIP_GROW px further than it is found (its faint soft edge), and
# the strip's shading read over STRIP_REACH px either side of it, from rows holding at least STRIP_MIN[0] pixels clear
# of print and columns holding STRIP_MIN[1], the column shading averaged over STRIP_SMOOTH px.
STRIP_GROW, STRIP_REACH, STRIP_MIN, STRIP_SMOOTH, STRIP_ROUNDS = 1, 60, (6, 3), 5, 5
RING_SHADE = 4       # px: the ring's soft shadow on the card, past its cut-out
BORDER_X = 26        # px: the plain silver card border on the left, which the Stage ring overlaps


def is_modern(card_id):
    return isinstance(card_id, str) and card_id.startswith(MODERN)


def read_card(card_id):
    """A Scarlet & Violet / Mega Evolution Pokemon in the shape the composers take, plus 'family' (regular | ex)."""
    c = tcg.card(card_id)
    name = c.get("name", "")
    if c.get("category") != "Pokemon":
        raise SystemExit(f"{card_id} is {name!r}: only Pokemon cards can be edited")
    if c.get("stage") not in ("Basic", "Stage1", "Stage2"):
        raise SystemExit(f"{card_id} is {name!r} ({c.get('stage')}): only Basic, Stage 1 and Stage 2 Pokemon")
    ex = name.endswith(" ex")
    if not ex and name.endswith(mecompose.NOT_REGULAR):
        raise SystemExit(f"{card_id} is {name!r}: only regular Pokemon and Pokemon ex")
    types = c.get("types") or []
    if not types or types[0] not in meframe.TYPES:
        raise SystemExit(f"{card_id}: type {types[:1]} is not supported")
    base = name[:-3] if ex else name
    prefix = next((p for p in compose.PREFIXES if base.startswith(p + " ")), "")
    for b in (c.get("abilities") or []) + (c.get("attacks") or []):
        b["effect"] = edit.printed(b.get("effect"))
        b["name"] = edit.printed(b.get("name"))
    evolves = edit.printed(c.get("evolveFrom")) if c["stage"] != "Basic" else ""
    card = {
        "id": card_id, "family": "ex" if ex else "regular", "type": types[0], "prefix": prefix,
        "name": base[len(prefix):].strip(), "hp": str(c.get("hp", "")),
        "stage": {"Basic": 0, "Stage1": 1, "Stage2": 2}[c["stage"]],
        "evolves_from": evolves, "portrait": "", "portrait_of": evolves or None,
        "abilities": c.get("abilities") or [], "attacks": c.get("attacks") or [],
        "weakness": (c.get("weaknesses") or [None])[0], "resistance": (c.get("resistances") or [None])[0],
        "retreat": int(c.get("retreat") or 0),
        # not drawn (the print keeps its own footer); carried for the ptcg-sim export (export.py)
        "illustrator": c.get("illustrator") or "", "reg_mark": c.get("regulationMark") or "",
        "rarity": mecompose.rarity_symbol(c.get("rarity")),
    }
    if ex:              # a Mega ex can change type on its own print (edit.retypes)
        card["retype"] = megaretype.supports(card_id, card)
    return card


def changed(original, card):
    """Which of GROUPS the edits change."""
    a, b = edit.editable(original), edit.editable(card)
    return tuple(g for g, keys in GROUPS.items() if any(a.get(k) != b.get(k) for k in keys))


# ---------------------------------------------------------------- drawing, the same way for finding and setting
def _name(canvas, card, logo=None):
    prof = frame.profile(card["type"])
    if card["family"] == "ex":
        return compose.name_bar_text(canvas, card, prof, ex=True, logo=logo)
    return compose.name_bar_text(canvas, card, prof, ex=False, name_right_limit=meframe.NAME_RIGHT_LIMIT)


def _moves(canvas, card, fill, edge=None, symbol=None):
    if card["family"] == "ex":
        return compose.text_block(canvas, card, fill, edge, symbol=symbol)
    return compose.text_block(canvas, card, fill, edge, layout=mecompose.me_layout,
                              name_to_body=mecompose.NAME_TO_BODY, fit_tracking=mecompose.BODY_FIT_TRACK,
                              min_tracking=mecompose.BODY_MIN_TRACK,
                              cost_r=mecompose.COST_R, cost_dy=mecompose.COST_DY, symbol=symbol)


def _row(canvas, card, fill, edge=None):
    icons = compose.SV_ROW_ICONS if card["family"] == "ex" else mecompose.ROW_ICONS
    compose.weakness_row(canvas, card, fill, edge, labels=False, icons=icons, retreat=card.get("retreat_span"))


def _row_sides(original, card):
    a, b = edit.editable(original), edit.editable(card)
    return [s for s in ("weakness", "resistance") if a.get(s) != b.get(s)]


def row_off(original, card):
    """The weakness-row print an edit takes off, as a card for _row: the sides that change and the Retreat circles a
    lower cost drops; what stays the same keeps its print. A side TCGdex leaves out (it misses some printed
    weaknesses) is looked for where the edit sets it."""
    sides = _row_sides(original, card)
    out = {**original, **{s: (original[s] or card[s]) if s in sides else None for s in ("weakness", "resistance")}}
    out["retreat_span"] = (min(card["retreat"], original["retreat"]), original["retreat"])
    return out


def row_on(original, card):
    """What the edit sets in the weakness row, as a card for _row: the sides that change and the circles a higher
    Retreat cost adds."""
    sides = _row_sides(original, card)
    out = {**card, **{s: card[s] if s in sides else None for s in ("weakness", "resistance")}}
    out["retreat_span"] = (min(card["retreat"], original["retreat"]), card["retreat"])
    return out


def _evolves(canvas, card):
    stage.evolves_line(canvas, card["evolves_from"], min(stage.EVOLVES_RIGHT, stage.strip_end(card) - stage.STRIP_PAD))


def _drawn(draw, card, *args):
    with layers.recording() as drawn:
        draw(np.zeros((H, W, 4), np.float32), card, *args)
    return drawn


def _cover(drawn, kinds):
    out = np.zeros((H, W), np.float32)
    for k in kinds:
        if k in drawn:
            np.maximum(out, drawn[k], out=out)
    return out


# ---------------------------------------------------------------- finding what is printed
def moves_off(scan, original):
    """What taking the old abilities and attacks off takes: everything in the text zone that stands out from the panel
    (moves_ink); on a regular card whose panel is textured (Ancient and Future prints: rock, rows of lit dots), only
    what stands out round where the old block is printed (moves_as_printed): the texture is the panel, and LaMa does
    not draw it again over the whole zone (sv05-061's dots came back a dark smear). -> bool HxW"""
    ink = moves_ink(scan, original["family"])
    if original["family"] != "regular":
        return ink
    moved = moves_as_printed(scan, original)
    cover = (_cover(moved, tuple(moved)) > 0.05).astype(np.uint8)
    near = dilate(cover > 0, PRINT_NEAR)
    x0, y0, x1, y1 = MOVES_ZONE[original["family"]]
    if (ink & ~near).sum() < TEXTURED * (x1 - x0) * (y1 - y0):
        return ink
    # along each line a word further: TCGdex does not always break lines where the print does, so a printed line
    # can run on past where the composers end it (sv05-109's "pile.", sv05-079's "Pokémon.)")
    along = cv2.dilate(cover, np.ones((2 * PRINT_NEAR + 1, 2 * LINE_RUN + 1), np.uint8)) > 0
    return ink & (near | along)


def moves_ink(scan, family):
    """The old ability/attack block: what stands out from the panel in the text zone. -> bool HxW"""
    return ink_in(scan, MOVES_ZONE[family])


def moves_as_printed(scan, original):
    """The composers' block of the original's abilities and attacks (_drawn of _moves: coverage by kind) moved onto
    its print: each move up or down to where the print's ink lies under it, then each of its lines a little further.
    Set where the composers set it, the old cost icons and badge would be looked for where the print has none
    (me02.5-152: its Ability lies 53 px above, its attack 23)."""
    drawn = _drawn(_moves, original, compose.WHITE)
    cover = dilate(_cover(drawn, tuple(drawn)) > 0.3, 1)        # the print's letters are not quite the composers'
    lines = _lines(cover)
    if not lines:
        return drawn
    strokes = ink_in(scan, MOVES_ZONE[original["family"]], holes=False)
    header = _cover(drawn, HEADERS) > 0.3
    out = {k: np.zeros((H, W), np.float32) for k in drawn}
    floor = -MOVE_SHIFT
    for move in _moves_of(lines, header):
        top, bottom = move[0][0], move[-1][1]
        shifts = range(max(-MOVE_SHIFT, floor - top), MOVE_SHIFT + 1)       # moves stay in their order
        dy = _best_shift(cover, strokes, top, bottom, shifts, 0)
        for a, b in move:
            dl = _best_shift(cover, strokes, a, b, range(dy - LINE_SHIFT, dy + LINE_SHIFT + 1), dy)
            for kind, v in drawn.items():
                _move_rows(out[kind], v, a, b, dl)
        floor = bottom + dy
    return out


def old_text(scan, original):
    """bool HxW: where the original's ability and attack text is printed (moves_as_printed), a little grown: the
    print's letters are not quite the composers'."""
    return dilate(moves_as_printed(scan, original).get("text", np.zeros((H, W), np.float32)) > 0.3, TEXT_REACH)


def _lines(cover):
    """[(first row, row past the last)] of the lines of a block's coverage, top to bottom."""
    rows = np.flatnonzero(cover.any(axis=1))
    if not rows.size:
        return []
    breaks = np.flatnonzero(np.diff(rows) > LINE_JOIN + 1)
    starts = np.concatenate([[rows[0]], rows[breaks + 1]])
    ends = np.concatenate([rows[breaks], [rows[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _moves_of(lines, header):
    """lines grouped into moves: a move starts at a line with a header kind (badge, cost icons) or after a gap of
    MOVE_GAP (an attack without cost)."""
    moves = []
    for a, b in lines:
        if not moves or header[a:b].any() or a - moves[-1][-1][1] >= MOVE_GAP:
            moves.append([])
        moves[-1].append((a, b))
    return moves


def _best_shift(cover, strokes, a, b, shifts, centre):
    """The shift of cover's rows a..b that lands most of it on the print's strokes (the nearest to centre of the
    best)."""
    part = cover[a:b]
    total = max(1, int(part.sum()))
    best, score = centre, -1.0
    for dy in sorted(shifts, key=lambda d: abs(d - centre)):
        if a + dy < 0 or b + dy > H:
            continue
        s = int((part & strokes[a + dy:b + dy]).sum()) / total
        if s > score:
            best, score = dy, s
    return best


def _move_rows(out, img, a, b, dy):
    """img's rows a..b set dy rows further down in out (kept where they overlap)."""
    a2, b2 = max(0, a + dy), min(H, b + dy)
    if a2 < b2:
        np.maximum(out[a2:b2], img[a2 - dy:b2 - dy], out=out[a2:b2])


def ink_in(scan, box, holes=True):
    """What stands out from the background inside box (x0, y0, x1, y1); holes: with what it encloses (icon
    middles). -> bool HxW"""
    x0, y0, x1, y1 = box
    rgb = np.clip(scan[..., :3], 0, 255).astype(np.uint8)
    pad = MEDIAN
    crop = cv2.copyMakeBorder(rgb[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_REFLECT)
    background = cv2.medianBlur(crop, MEDIAN)
    diff = np.abs(crop.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    ink = (diff > INK_DIFF).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((INK_CLOSE, INK_CLOSE), np.uint8))
    ink = (fill_holes(ink > 0) if holes else ink > 0)[pad:-pad, pad:-pad]
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = ink
    return out


def ink_near(scan, cover, reach=REACH, holes=True):
    """Where something set by the composers is really printed: the print that stands out near it, each shape of it
    whole if it ends within WHOLE_REACH (the print can sit further off than ours, sv04-128's "-30"), plus the set
    shapes themselves (holes=False: the print only). cover: HxW coverage. -> bool HxW"""
    near = dilate(cover > 0.05, reach)
    if not near.any():
        return near
    far = dilate(cover > 0.05, reach * WHOLE_REACH)
    ys, xs = np.nonzero(far)
    m = MEDIAN          # the background estimate needs room around the print, or the print is its own background
    box = (max(0, xs.min() - m), max(0, ys.min() - m), min(W, xs.max() + m), min(H, ys.max() + m))
    ink = ink_in(scan, box, holes) & near
    strokes = ink_in(scan, box, holes=False)          # holes would join a bar's two rules into one band
    rules = cv2.morphologyEx(strokes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, RULE), np.uint8)) > 0
    strokes &= ~dilate(rules, 1)                      # glyphs touching a rule are their own shapes
    _, labels = cv2.connectedComponents(strokes.astype(np.uint8))
    spills = np.unique(labels[strokes & ~far])        # what runs on past the print is not it: only its near part
    whole = np.setdiff1d(np.unique(labels[strokes & near]), np.append(spills, 0))
    found = ink | np.isin(labels, whole)
    return found | (cover > 0.05) if holes else found


def _glyphs(ink):
    """bool HxW: the shapes of ink that sit in lines of text."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    sized = [i for i in range(1, n) if GLYPH_H[0] <= stats[i, cv2.CC_STAT_HEIGHT] <= GLYPH_H[1]
             and stats[i, cv2.CC_STAT_WIDTH] <= GLYPH_W and stats[i, cv2.CC_STAT_AREA] >= GLYPH_MIN]
    bottoms = np.array([stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in sized])
    lined = [i for i, b in zip(sized, bottoms) if (np.abs(bottoms - b) <= 1).sum() >= TEXT_LINE]
    return np.isin(labels, lined)


def _stock(rgb):
    """The stock ink a measured colour is, or the colour itself."""
    for stock in (compose.WHITE, np.array(frame.DARK_INK, np.float32)):
        if float(np.linalg.norm(rgb - stock)) <= INK_SNAP:
            return stock
    return rgb.astype(np.float32)


def _outlined(glyphs, lift, sign, off):
    """Whether the glyphs (darker than their panel: sign -1) sit in an outline of the other shade: one that stands
    out from the panel in colour (off: how far each pixel is from it), as white does on a yellow one (sv05-061)."""
    ring = dilate(glyphs, KEYLINE_RING) & ~glyphs
    return (bool(ring.any()) and -sign * float(np.median(lift[ring])) > 0
            and float(np.median(off[ring])) > INK_DIFF)


def print_ink(scan, box, where=None):
    """The ink text is printed in inside box: (fill RGB, keyline (RGB, radius, opacity) or None), or None when no
    lines of text are found. Dark and light print are read apart: a dark glyph's white outline (full art) would
    otherwise join it, and the art, into one shape. where (bool HxW): only shapes there are the text (where the old
    text is printed): a Future print's panel is lit dots in rows, which read as lines of light glyphs."""
    x0, y0, x1, y1 = box
    rgb = np.clip(scan[..., :3], 0, 255).astype(np.uint8)
    pad = MEDIAN
    crop = cv2.copyMakeBorder(rgb[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_REFLECT)
    background = cv2.medianBlur(crop, MEDIAN)[pad:-pad, pad:-pad].astype(np.float32)
    seen = crop[pad:-pad, pad:-pad].astype(np.float32)
    lift = luminance(seen) - luminance(background)
    dark, light = lift < -INK_DIFF, lift > INK_DIFF
    if where is not None:
        dark, light = dark & where[y0:y1, x0:x1], light & where[y0:y1, x0:x1]
    dark, light = _glyphs(dark), _glyphs(light)
    off = np.abs(seen - background).max(axis=-1)
    dark_outlined, light_outlined = _outlined(dark, lift, -1, off), _outlined(light, lift, 1, off)
    if dark_outlined != light_outlined:                     # an outline is never outlined itself: half its ring
        sign = -1 if dark_outlined else 1                   # is the panel
    else:
        sign = -1 if dark.sum() >= light.sum() else 1       # the print is darker (-1) or lighter than its panel
    glyphs, outlined = (dark, dark_outlined) if sign < 0 else (light, light_outlined)
    if not glyphs.any():
        return None
    strength = np.abs(lift)
    core = glyphs & (strength >= np.median(strength[glyphs]))      # past the glyphs' anti-aliased edges
    fill = _stock(seen[core].mean(axis=0))
    ring = dilate(glyphs, KEYLINE_RING) & ~glyphs
    edge = None
    if outlined:
        colour = _stock(np.median(seen[ring], axis=0))
        edge = (colour, keyline_width(glyphs, seen, colour), compose.LIGHT_KEYLINE[2])
    return fill, edge


def keyline_width(glyphs, seen, colour):
    """How far the print's outline reaches past its glyphs (px): the last ring a pixel wide round them that is mostly
    in the outline's colour, and the share of the next one that still is."""
    shares = []
    for r in range(1, KEYLINE_MAX + 1):
        ring = dilate(glyphs, r) & ~dilate(glyphs, r - 1)
        shares.append(float((np.abs(seen[ring] - colour).max(-1) < KEYLINE_NEAR).mean()) if ring.any() else 0.0)
    full = [r for r, share in enumerate(shares, 1) if share >= 0.5]
    if not full:
        return compose.LIGHT_KEYLINE[1]
    last = max(full)
    return last + (shares[last] if last < len(shares) else 0.0)


def ink_colour(scan, where):
    """White or dark: what the old text was printed in - lighter or darker than what it is printed on."""
    lum = luminance(scan[..., :3])
    around = dilate(where, 4) & ~dilate(where, 1)
    if not where.any() or not around.any():
        return compose.WHITE
    lighter = float(np.median(lum[where])) > float(np.median(lum[around]))
    return compose.WHITE if lighter else np.array(frame.DARK_INK, np.float32)


def card_logo(scan, original):
    """The card's own ex logo (Scarlet & Violet or Mega Evolution style), cut where the original name sets it,
    as (sprite, left x, right x) for compose.name_bar_text."""
    logo = _drawn(_name, original)["logo"]
    shape = dilate(logo > 0.2, 2)
    ys, xs = np.nonzero(shape)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = cv2.GaussianBlur(shape[y0:y1, x0:x1].astype(np.float32), (3, 3), 0.7)
    sprite = (scan[y0:y1, x0:x1, :3].copy(), alpha, x0, y0, (x0, y0))
    # name_bar_text places the logo by its left edge: frame.ex_logo's left/right are the drawn shape's columns
    cols = np.flatnonzero((logo > 0.5).any(axis=0))
    return sprite, int(cols[0]), int(cols[-1]) + 1


# ---------------------------------------------------------------- editing
def _stage_kit(card):
    return stage.REGULAR if card["family"] == "regular" else stage.EX


def stage_parts(original, card):
    """The Stage pieces an edited card has drawn (stage.draw's parts): all of them on a Basic made Stage, the tab
    for another stage or back to Basic, the strip for a new "Evolves from", the disc for a new portrait."""
    before, after = original["stage"], card["stage"]
    if not after:
        return ("tab",) if before else ()
    if not before:
        return _stage_kit(card)
    renamed = card["evolves_from"] != original["evolves_from"]
    return tuple(p for p, on in (("tab", after != before), ("strip", renamed),
                                 ("disc", renamed or card.get("portrait") != original.get("portrait"))) if on)


def stage_off(original, card):
    """bool HxW: the Stage pieces (and their shadow) a card made Basic loses; empty otherwise."""
    if not original["stage"] or card["stage"]:
        return np.zeros((H, W), bool)
    sprite = meframe.stage_sprite(original["stage"])
    pieces = (sprite[..., 3] > 25) & stage.part_mask(tuple(p for p in _stage_kit(original) if p != "tab"))
    return dilate(pieces | stage.under_tab(sprite), RING_SHADE)


def taken_off(card_id, original, groups, card=None, scan=None):
    """What the edited groups take off the card (scan: the print they are taken off, default the card's own). ->
    (bool HxW for LaMa, bool HxW filled along pixel rows: the weakness row and the "Evolves from" line)"""
    scan = tcg.scan(card_id) if scan is None else scan
    mask, row = np.zeros((H, W), bool), np.zeros((H, W), bool)
    if "stage" in groups and card is not None:
        mask |= stage_off(original, card)
    if "name" in groups:
        mask |= ink_near(scan, _cover(_drawn(_name, original), ("name", "hp", "logo")))
    if "moves" in groups:
        mask |= moves_off(scan, original)
    if "row" in groups:
        # tight: the weakness icon sits close to the end of its label
        printed_row = _drawn(_row, row_off(original, card) if card is not None else original, compose.WHITE)
        row = dilate(ink_near(scan, _cover(printed_row, ("row_icon", "row_text")), ROW_NEAR), 1)
        # the icons' soft drop shadow, too faint to stand out, falls on the bar below them
        icons = _cover(printed_row, ("row_icon",)) > 0.1
        row |= dilate(icons | np.roll(icons, ICON_SHADOW, axis=0), ICON_SHADOW)
    if ("stage" in groups and card is not None and original["stage"] and card["stage"]
            and card["evolves_from"] != original["evolves_from"]):
        # on the strip, and the old line's own letters off it: what stands out round them below the strip is the art
        old = _cover(_drawn(_evolves, original), ("evolves",))
        row |= dilate(ink_near(scan, old, ROW_NEAR), 1) & (stage.strip_shape(scan) | dilate(old > 0.05, 1))
    return dilate(mask, MARK_GROW), row


def fill_band(img, where):
    """Fill the `where` pixels of the weakness row along each pixel row: every gap blends from the colour just left of
    it to the colour just right of it (BAND_SAMPLE px a little way out, skipping print). The row's lines run across,
    so this keeps them whole; LaMa would copy the labels into the gap, 2-D inpainting draws crosses. Each pixel's
    blend off the bars' rules is the median of the BAND_SMOOTH rows' about it within BAND_STEP of most of them (a
    row's end colours are a few pixels each: alone they streak, or catch a shadow). The
    blend then takes on the band's grain: each BAND_PIECE px square piece of a gap copies the grain of the same
    shape of band beside it, the nearest that holds no print (the bars' rules and the band's own specks are band)."""
    out = img.copy()
    rows = np.flatnonzero(where.any(axis=1))
    if not len(rows):
        return out
    printed, rules = _strong_print(img, rows[0], rows[-1] + 1)
    skip = where | printed
    pieces, shifts = _band_pieces(where, skip)
    blend = np.full(img.shape[:2] + (3,), np.nan, np.float32)
    for y in rows:
        for x0, x1 in _runs_of(where[y]):
            left = _sample(img[y], skip[y], x0 - 1, -1)
            right = _sample(img[y], skip[y], x1, 1)
            if left is None and right is None:
                continue
            left = right if left is None else left
            right = left if right is None else right
            t = (np.arange(x0, x1) - x0 + 0.5)[:, None] / max(1, x1 - x0)
            blend[y, x0:x1] = left * (1 - t) + right * t
    y0, y1 = rows[0], rows[-1] + 1
    band = blend[y0:y1].copy()
    band[rules[y0:y1]] = np.nan                     # a rule's rows are each their own: not smoothed
    half = BAND_SMOOTH // 2
    padded = np.pad(band, ((half, half), (0, 0), (0, 0)), constant_values=np.nan)
    near = np.stack([padded[i:i + y1 - y0] for i in range(BAND_SMOOTH)])
    with warnings.catch_warnings():                 # pixels with no band rows about them: stay blended
        warnings.simplefilter("ignore", RuntimeWarning)
        smooth = _like(near, np.nanmedian(near, axis=0))
    filled = ~np.isnan(blend[y0:y1, :, 0])
    out[y0:y1, :, :3][filled] = blend[y0:y1][filled]
    steady = filled & ~np.isnan(band[..., 0]) & ~np.isnan(smooth[..., 0])
    out[y0:y1, :, :3][steady] = smooth[steady]
    for y in rows:
        for x0, x1 in _runs_of(where[y]):
            for a, b in _runs(pieces[y, x0:x1]):
                d = shifts.get(pieces[y, x0 + a])
                if d is not None and not np.isnan(blend[y, x0 + a, 0]):
                    grain = img[y, x0 + a + d:x0 + b + d, :3]
                    out[y, x0 + a:x0 + b, :3] += grain - grain.mean(axis=0)
    return out


def fill_strip(img, where):
    """Fill the `where` pixels of the "Evolves from" strip (its old line taken off) with the strip's own shading. The
    strip is lit in bands that run along it (a highlight, the body, a darker lower edge), each a little lighter or
    darker from place to place along it: so a pixel is its row's shade plus its column's, both read off the strip
    about the line where it is clear of print, as stage.blank_tab reads the tab's. (Blending along the rows, as
    fill_band does, takes the ring's rim left of the line and the letters' soft edges between its words.) The fill
    then takes on the strip's grain, each piece from the nearest clean stretch of strip, as fill_band's gaps do."""
    out = img.copy()
    where = dilate(where, STRIP_GROW) & stage.strip_shape(img)
    ys, xs = np.nonzero(where)
    if not len(ys):
        return out
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = max(stage.STRIP[0], xs.min() - STRIP_REACH), min(stage.STRIP[2], xs.max() + 1 + STRIP_REACH)
    printed, _ = _strong_print(img, y0, y1)
    skip = where | printed
    seen = img[y0:y1, x0:x1, :3].astype(np.float32)
    clean = ~skip[y0:y1, x0:x1]
    values = np.where(clean[..., None], seen, np.nan)
    by_column = np.zeros((x1 - x0, 3), np.float32)
    with warnings.catch_warnings():                 # rows / columns with no clean pixels: filled after
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(STRIP_ROUNDS):
            by_row = stage.fill_gaps(np.nanmedian(values - by_column[None], axis=1), clean.sum(1) >= STRIP_MIN[0])
            by_column = stage.fill_gaps(np.nanmedian(values - by_row[:, None], axis=0), clean.sum(0) >= STRIP_MIN[1])
            by_column = cv2.blur(by_column[None], (STRIP_SMOOTH, 1))[0]
    shade = np.full(img.shape[:2] + (3,), np.nan, np.float32)
    shade[y0:y1, x0:x1] = by_row[:, None] + by_column[None]
    out[where, :3] = shade[where]
    grain = np.where(skip[..., None], np.nan, img[..., :3] - shade)
    pieces, shifts = _band_pieces(where, skip)
    for y in range(y0, y1):
        for a, b in _runs_of(where[y]):
            for p, q in _runs(pieces[y, a:b]):
                d = shifts.get(pieces[y, a + p])
                if d is None:
                    continue
                piece = grain[y, a + p + d:a + q + d]
                if not np.isnan(piece).any():
                    out[y, a + p:a + q, :3] += piece - piece.mean(axis=0)
    return out


def _like(near, centre):
    """Per pixel, the median of the rows in near (rows x H x W x 3) within BAND_STEP of centre (H x W x 3)."""
    kept = near.copy()
    kept[np.repeat(np.abs(near - centre[None]).max(axis=3, keepdims=True) > BAND_STEP, 3, axis=3)] = np.nan
    return np.nanmedian(kept, axis=0)


def _runs_of(line):
    """(start, end) of each stretch of True in a bool line."""
    row = line.astype(np.int8)
    return zip(np.flatnonzero(np.diff(np.r_[0, row]) == 1), np.flatnonzero(np.diff(np.r_[row, 0]) == -1) + 1)


def _strong_print(img, y0, y1):
    """(bool HxW: print about rows y0..y1 that stands out from the band more than its grain does (PRINT_DIFF), icons
    whole, grown by its soft edge; bool HxW: the bars' rules, which run across and are band, not print)."""
    top, bottom = max(0, y0 - MEDIAN), min(img.shape[0], y1 + MEDIAN)
    crop = np.clip(img[top:bottom, :, :3], 0, 255).astype(np.uint8)
    lift = (np.abs(crop.astype(np.int16) - cv2.medianBlur(crop, MEDIAN).astype(np.int16)).max(axis=2)
            > PRINT_DIFF).astype(np.uint8)
    ruled = cv2.morphologyEx(lift, cv2.MORPH_OPEN, np.ones((1, RULE), np.uint8))       # rules and their shading
    lift = cv2.morphologyEx(lift & ~ruled, cv2.MORPH_CLOSE, np.ones((INK_CLOSE, INK_CLOSE), np.uint8)) > 0
    printed, rules = np.zeros(img.shape[:2], bool), np.zeros(img.shape[:2], bool)
    printed[top:bottom] = dilate(fill_holes(lift), PRINT_GROW)
    rules[top:bottom] = ruled > 0
    return printed, rules


def _runs(ids):
    """(start, end) of each stretch of equal values in ids."""
    cuts = np.flatnonzero(np.diff(ids)) + 1
    return list(zip(np.r_[0, cuts], np.r_[cuts, len(ids)]))


def _band_pieces(where, skip):
    """(int HxW: where cut into pieces BAND_PIECE px square, numbered from 1; {piece: x shift to the nearest place
    of the same shape clear of skip, within BAND_SEARCH px}). Pieces with no such place have no shift."""
    n, comps = cv2.connectedComponents(where.astype(np.uint8))
    pieces = np.zeros(where.shape, np.int32)
    shifts = {}
    for c in range(1, n):
        ys, xs = np.nonzero(comps == c)
        part = (xs - xs.min()) // BAND_PIECE * H + (ys - ys.min()) // BAND_PIECE
        for p in np.unique(part):
            key = len(shifts) + 1
            py, px = ys[part == p], xs[part == p]
            pieces[py, px] = key
            shifts[key] = None
            least = px.max() - px.min() + 1 + BAND_GAP
            for extra in range(BAND_SEARCH):
                d = next((d for d in (-(least + extra), least + extra)
                          if px.min() + d >= 0 and px.max() + d < where.shape[1] and not skip[py, px + d].any()), None)
                if d is not None:
                    shifts[key] = d
                    break
    return pieces, shifts


def _sample(line, skip, x, step):
    """Median colour of BAND_SAMPLE unprinted pixels going from x in direction step, after BAND_GAP px and within
    BAND_REACH px past that (beyond, the band may be the card's border); None when there are none."""
    picked, x = [], x + step * BAND_GAP
    stop = x + step * BAND_REACH
    while 0 <= x < len(line) and x != stop and len(picked) < BAND_SAMPLE:
        if not skip[x]:
            picked.append(line[x, :3])
        x += step
    return np.median(picked, axis=0) if picked else None


def ghosts(art, mask):
    """bool HxW: what stands out in the inpainted area the way printed ink does, in blots on a plain panel: LaMa's
    inventions. Nothing printed is left there (it was all taken off), so on a plain panel nothing should stand out;
    on art (a full-art card's text zone) standing out is the art, and is left alone."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((H, W), bool)
    standing = ink_in(art, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)) & mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(standing.astype(np.uint8))
    lum = luminance(art[..., :3])
    out = np.zeros((H, W), bool)
    for i in range(1, n):
        if not GHOST_MIN <= stats[i, cv2.CC_STAT_AREA] <= GHOST_MAX:
            continue
        blot = labels == i
        ring = dilate(blot, GHOST_RING[1]) & ~dilate(blot, GHOST_RING[0])
        if ring.any() and float(lum[ring].std()) < GHOST_PLAIN:
            out |= blot
    return out


def plain_panel(card_id, original):
    """True when the text zone is a plain panel. A Pokemon ex's text box lies over its art, and an illustration
    rare's art runs under the text, so standouts there are the art and ghosts() must not read them as blots."""
    rarity = (tcg.card(card_id).get("rarity") or "").lower()
    return original["family"] == "regular" and "illustration" not in rarity


def patch_blots(art, blots, mask):
    """Cover each blot with a patch of the plain panel beside it (LaMa, asked again, leaves a rim): the nearest
    clean patch of the same shape, its light matched to the panel round the blot, the edge feathered."""
    out = art.copy()
    lum = luminance(art[..., :3])
    ys, xs = np.nonzero(mask)
    busy = ink_in(art, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)) | blots
    n, labels = cv2.connectedComponents(blots.astype(np.uint8))
    for i in range(1, n):
        hole = dilate(labels == i, GHOST_GROW)
        hy, hx = np.nonzero(hole)
        y0, y1, x0, x1 = hy.min(), hy.max() + 1, hx.min(), hx.max() + 1
        size = max(y1 - y0, x1 - x0)
        ring = dilate(hole, GHOST_RING[1]) & ~hole
        best = None
        for step in (1, 2, 3):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                oy, ox = dy * (size + 4) * step, dx * (size + 4) * step
                if not (0 <= y0 + oy and y1 + oy <= H and 0 <= x0 + ox and x1 + ox <= W):
                    continue
                src = np.roll(hole, (oy, ox), axis=(0, 1))
                if (src & busy).any():
                    continue
                spread = float(lum[src].std())
                if best is None or spread < best[0]:
                    best = (spread, oy, ox)
            if best:
                break
        if best is None:
            continue
        _, oy, ox = best
        patch = np.roll(art[..., :3], (-oy, -ox), axis=(0, 1))       # the source moved onto the hole
        src_ring = np.roll(ring, (oy, ox), axis=(0, 1))
        patch = patch + (art[ring, :3].mean(axis=0) - art[src_ring, :3].mean(axis=0))
        a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.5)[..., None]
        a[~dilate(hole, 3)] = 0
        out[..., :3] = out[..., :3] * (1 - a) + patch * a
    return out


def cleaned(card_id, original, groups, card=None, scan=None):
    """The print (scan, default the card's own) with the edited groups taken off; cached by the print and what is
    taken off (a Mega ex recoloured is another print). -> (RGBA float32, mask taken off)"""
    scan = tcg.scan(card_id) if scan is None else scan
    mask, row = taken_off(card_id, original, groups, card, scan)
    key = hashlib.sha256(np.ascontiguousarray(scan, np.float32).tobytes() + mask.tobytes()
                         + row.tobytes()).hexdigest()[:12]
    out = cache.path("art", f"modern-v{CLEAN_VERSION}-{card_id}-{'-'.join(groups)}-{key}.png")
    if out.exists():
        return load_rgba(out), mask | row
    art = inpaint.inpaint(scan, mask) if mask.any() else scan.copy()
    blots = ghosts(art, mask) if plain_panel(card_id, original) else None
    if blots is not None and blots.any():
        art = patch_blots(art, blots, mask)
    if row.any():
        line = row & stage.strip_shape(art)               # the old "Evolves from" line, on its strip
        art = fill_strip(fill_band(art, row & ~line), line)
    border = mask.copy()
    border[:, BORDER_X:] = False
    if border.any():                 # LaMa leaves blotches on the plain border: it shades only from top to bottom
        art = fill_band(art.transpose(1, 0, 2), border.T).transpose(1, 0, 2)
    save_rgba(art, out)
    return art, mask | row


def _row_colour(scan, original, card):
    """White or dark, for a weakness row print_ink finds no labels in: what its old values were printed in."""
    old = ink_near(scan, _cover(_drawn(_row, row_off(original, card), compose.WHITE), ("row_text",)), holes=False)
    return ink_colour(scan, old)


def _block_colour(scan, original):
    """White or dark, for a block print_ink finds no lines of text in."""
    strokes = ink_in(scan, MOVES_ZONE[original["family"]], holes=False)
    drawn = _cover(_drawn(_moves, original, compose.WHITE), ("icon", "badge", "ability_name"))
    return ink_colour(scan, strokes & ~dilate(drawn > 0.1, 6))


def still_printed(original, card, groups):
    """The card as it is still printed once the edited groups are taken off: what retype keeps of its print."""
    shown = dict(original)
    if "moves" in groups:
        shown.update(abilities=[], attacks=[])
    if "row" in groups:
        shown.update({side: None for side in _row_sides(original, card)})
        shown["retreat"] = min(original["retreat"], card["retreat"])
    if "stage" in groups and not card["stage"]:
        shown["stage"] = 0
    return shown


@functools.lru_cache(maxsize=4)
def _recoloured(card_id, type_name):
    """A Mega ex print recoloured on another type (megaretype.recolour) -> (RGBA, the words it re-inked)."""
    with layers.recording() as drawn:
        out = megaretype.recolour(tcg.scan(card_id), read_card(card_id), type_name)
    return out, drawn.get(megaretype.WORDS)


def print_of(card_id, original, card, record=True):
    """(the print the edits start from, the card as that print has it): the card's own scan; a Mega ex whose type
    changes, its print recoloured on the new type (megaretype), which is then the card as printed. record: the words
    it re-inked go to layers (megaretype.WORDS), for the plates."""
    scan = tcg.scan(card_id)
    if original["family"] != "ex" or not original.get("retype") or card["type"] == original["type"]:
        return scan, original
    out, words = _recoloured(card_id, card["type"])
    if record and words is not None:
        with layers.kind(megaretype.WORDS):
            layers.record(words)
    return out.copy(), dict(original, type=card["type"])


def compose_card(card_id, card, original=None):
    """-> (RGBA float32 733x1024, report). card: read_card's dict, edited; original: as printed (default: read)."""
    original = original or read_card(card_id)
    scan, printed_as = print_of(card_id, original, card)
    retyped = ("type",) if printed_as is not original else ()
    original = printed_as
    groups = changed(original, card)
    if not groups:
        return scan.copy(), {"card": card, "edited": list(retyped), "taken_off": None}
    plan = inplace.find(scan, original, card) if "moves" in groups else None
    parts = tuple(g for g in groups if g not in ("type", "rule") and not (g == "moves" and plan))   # what is taken off
    if parts:
        canvas, mask = cleaned(card_id, original, parts, card, scan)
        canvas = canvas.copy()
    else:
        canvas, mask = scan.copy(), np.zeros((H, W), bool)
    if "type" in groups:
        canvas = retype.recolour(canvas, still_printed(original, card, parts), card["type"])
    if "stage" in groups:
        own = lambda: stage.disc_sprite(card_id)
        printed = bool(original["stage"])
        stage.draw(canvas, card, stage_parts(original, card), own if printed else None, printed=printed)
    if "name" in groups:
        logo = card_logo(scan, original) if card["family"] == "ex" else None
        _name(canvas, card, logo)
    if "rule" in groups and card.get("mega_rule"):
        megarule.draw(canvas, card, printed_badge=megarule.prints_badge(original))
    if plan:
        canvas = inplace.apply(canvas, plan, original)
    elif "moves" in groups:
        fill, edge = ((retype.ink(card["type"]), None) if "type" in groups else
                      print_ink(scan, MOVES_ZONE[original["family"]], old_text(scan, original))
                      or (_block_colour(scan, original), None))
        shrink = _moves(canvas, card, fill, edge)
        if shrink is not None and shrink < 1:
            warnings.warn(f"{card_id}: ability/attack text set at {shrink:.0%} to fit")
    if "row" in groups:
        fill, edge = ((retype.ink(card["type"]), None) if "type" in groups else
                      print_ink(scan, ROW_LABELS) or (_row_colour(scan, original, card), None))
        _row(canvas, row_on(original, card), fill, edge)
    canvas[..., 3] = scan[..., 3]
    return canvas, {"card": card, "edited": list(retyped + groups), "taken_off": mask}
