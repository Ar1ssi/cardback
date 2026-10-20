"""Edit Scarlet & Violet and Mega Evolution cards in place: the card keeps its own scan, frame and art; only what
was edited is taken off and set again in the same layouts the restyles use.

- Name and HP, and the weakness row (weakness, resistance, retreat): the original is set again with the composers
  (they match these prints to a pixel or two, see tests) to find exactly where it is printed.
- Stage, "Evolves from" and the evolution portrait: the Stage pieces sit at one place on every SV / ME card, so the
  ones that changed are pasted over (stage.py); back to Basic they are taken off and the BASIC tab set.
- Abilities and attacks: ex prints do not stack their blocks by one rule, so the old block is found on the card
  instead: everything in the text zone that stands out from the panel behind it (text, cost icons, the Ability
  badge). Regular cards then take the new block in the Mega Evolution layout (Scarlet & Violet regular cards share
  it), ex cards in the Absol ex layout.
What is taken off is rebuilt by LaMa (cached per card and set of edited parts), except in the weakness row, which
is filled along its pixel rows (fill_band). Energy written {G}, {R} ... in effect text prints as its symbol.
"""
import hashlib
import warnings

import cv2
import numpy as np

from . import cache, compose, edit, frame, inpaint, layers, mecompose, meframe, stage, tcg
from .imaging import dilate, fill_holes, load_rgba, luminance, save_rgba

W, H = 733, 1024
MODERN = ("sv", "me")
GROUPS = {"name": ("prefix", "name", "hp"), "moves": ("abilities", "attacks"),
          "row": ("weakness", "resistance", "retreat"), "stage": ("stage", "evolves_from", "portrait")}

# Abilities and attacks sit in this zone: x0, y0, x1, y1 (under the window bar / Pokedex strip, above the weakness
# row).
MOVES_ZONE = {"regular": (36, 512, 697, 868), "ex": (30, 524, 703, 870)}
MEDIAN = 45          # px: wider than a glyph or cost icon, so the median is the panel behind them
INK_DIFF = 38        # how far a pixel stands out from that panel to count as printed
INK_CLOSE = 5        # px: joins glyph strokes and icon rims into whole shapes
MARK_GROW = 3        # px around everything taken off (anti-aliasing, keylines)
REACH = 5            # px: name, HP and weakness-row print may sit this far from where the composers set them
ROW_NEAR = 4         # REACH for the weakness row
BAND_GAP, BAND_SAMPLE = 3, 6   # px: weakness-row gaps take their colour this far out, from this many pixels
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
    if not types or types[0] not in meframe.POOL:
        raise SystemExit(f"{card_id}: type {types[:1]} is not supported")
    base = name[:-3] if ex else name
    prefix = next((p for p in compose.PREFIXES if base.startswith(p + " ")), "")
    for b in (c.get("abilities") or []) + (c.get("attacks") or []):
        b["effect"] = edit.printed(b.get("effect"))
        b["name"] = edit.printed(b.get("name"))
    evolves = edit.printed(c.get("evolveFrom")) if c["stage"] != "Basic" else ""
    return {
        "id": card_id, "family": "ex" if ex else "regular", "type": types[0], "prefix": prefix,
        "name": base[len(prefix):].strip(), "hp": str(c.get("hp", "")),
        "stage": {"Basic": 0, "Stage1": 1, "Stage2": 2}[c["stage"]],
        "evolves_from": evolves, "portrait": "", "portrait_of": evolves or None,
        "abilities": c.get("abilities") or [], "attacks": c.get("attacks") or [],
        "weakness": (c.get("weaknesses") or [None])[0], "resistance": (c.get("resistances") or [None])[0],
        "retreat": int(c.get("retreat") or 0),
    }


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


def _moves(canvas, card, fill):
    if card["family"] == "ex":
        return compose.text_block(canvas, card, fill, None)
    return compose.text_block(canvas, card, fill, None, layout=mecompose.me_layout,
                              name_to_body=mecompose.NAME_TO_BODY, fit_tracking=mecompose.BODY_FIT_TRACK,
                              cost_r=mecompose.COST_R, cost_dy=mecompose.COST_DY)


def _row(canvas, card, fill):
    icons = compose.SV_ROW_ICONS if card["family"] == "ex" else mecompose.ROW_ICONS
    compose.weakness_row(canvas, card, fill, None, labels=False, icons=icons)


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
def moves_ink(scan, family):
    """The old ability/attack block: what stands out from the panel in the text zone. -> bool HxW"""
    return ink_in(scan, MOVES_ZONE[family])


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
    """Where something set by the composers is really printed: the print that stands out near it, plus the set
    shapes themselves (holes=False: the print only). cover: HxW coverage. -> bool HxW"""
    near = dilate(cover > 0.05, reach)
    if not near.any():
        return near
    ys, xs = np.nonzero(near)
    m = MEDIAN          # the background estimate needs room around the print, or the print is its own background
    box = (max(0, xs.min() - m), max(0, ys.min() - m), min(W, xs.max() + m), min(H, ys.max() + m))
    found = ink_in(scan, box, holes) & near
    return found | (cover > 0.05) if holes else found


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


def taken_off(card_id, original, groups, card=None):
    """What the edited groups take off the card. -> (bool HxW for LaMa, bool HxW for the weakness row)"""
    scan = tcg.scan(card_id)
    mask, row = np.zeros((H, W), bool), np.zeros((H, W), bool)
    if "stage" in groups and card is not None:
        mask |= stage_off(original, card)
    if "name" in groups:
        mask |= ink_near(scan, _cover(_drawn(_name, original), ("name", "hp", "logo")))
    if "moves" in groups:
        mask |= moves_ink(scan, original["family"])
    if "row" in groups:
        # tight: the weakness icon sits close to the end of its label
        row = dilate(ink_near(scan, _cover(_drawn(_row, original, compose.WHITE), ("row_icon", "row_text")),
                              ROW_NEAR), 1)
    return dilate(mask, MARK_GROW), row


def fill_band(img, where, keep_out):
    """Fill the `where` pixels of the weakness row along each pixel row: every gap blends from the colour just left of
    it to the colour just right of it (BAND_SAMPLE px a little way out, skipping print). The row's lines run across,
    so this keeps them whole; LaMa would copy the labels into the gap, 2-D inpainting draws crosses."""
    out = img.copy()
    skip = where | keep_out
    for y in np.flatnonzero(where.any(axis=1)):
        row = where[y].astype(np.int8)
        starts = np.flatnonzero(np.diff(np.r_[0, row]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[row, 0]) == -1) + 1
        for x0, x1 in zip(starts, ends):
            left = _sample(img[y], skip[y], x0 - 1, -1)
            right = _sample(img[y], skip[y], x1, 1)
            if left is None and right is None:
                continue
            left = right if left is None else left
            right = left if right is None else right
            t = (np.arange(x0, x1) - x0 + 0.5)[:, None] / max(1, x1 - x0)
            out[y, x0:x1, :3] = left * (1 - t) + right * t
    return out


def _sample(line, skip, x, step):
    """Median colour of BAND_SAMPLE unprinted pixels going from x in direction step, after BAND_GAP px."""
    picked, x = [], x + step * BAND_GAP
    while 0 <= x < len(line) and len(picked) < BAND_SAMPLE:
        if not skip[x]:
            picked.append(line[x, :3])
        x += step
    return np.median(picked, axis=0) if picked else None


def cleaned(card_id, original, groups, card=None):
    """The scan with the edited groups taken off (cached). -> (RGBA float32, mask taken off)"""
    mask, row = taken_off(card_id, original, groups, card)
    key = hashlib.sha256(mask.tobytes() + row.tobytes()).hexdigest()[:12]
    out = cache.path("art", f"modern-{card_id}-{'-'.join(groups)}-{key}.png")
    if out.exists():
        return load_rgba(out), mask | row
    scan = tcg.scan(card_id)
    art = inpaint.inpaint(scan, mask) if mask.any() else scan.copy()
    if row.any():
        art = fill_band(art, row, ink_in(scan, (0, 860, W, 930), holes=False))
    border = mask.copy()
    border[:, BORDER_X:] = False
    if border.any():                 # LaMa leaves blotches on the plain border: it shades only from top to bottom
        art = fill_band(art.transpose(1, 0, 2), border.T, np.zeros((W, H), bool)).transpose(1, 0, 2)
    save_rgba(art, out)
    return art, mask | row


def compose_card(card_id, card, original=None):
    """-> (RGBA float32 733x1024, report). card: read_card's dict, edited; original: as printed (default: read)."""
    original = original or read_card(card_id)
    groups = changed(original, card)
    scan = tcg.scan(card_id)
    if not groups:
        return scan.copy(), {"card": card, "edited": [], "taken_off": None}
    canvas, mask = cleaned(card_id, original, groups, card)
    canvas = canvas.copy()
    if "stage" in groups:
        own = lambda: stage.disc_sprite(card_id)
        stage.draw(canvas, card, stage_parts(original, card), own if original["stage"] else None)
    if "name" in groups:
        logo = card_logo(scan, original) if card["family"] == "ex" else None
        _name(canvas, card, logo)
    if "moves" in groups:
        strokes = ink_in(scan, MOVES_ZONE[original["family"]], holes=False)
        fill = ink_colour(scan, strokes & ~dilate(_cover(_drawn(_moves, original, compose.WHITE),
                                                         ("icon", "badge", "ability_name")) > 0.1, 6))
        shrink = _moves(canvas, card, fill)
        if shrink is not None and shrink < 1:
            warnings.warn(f"{card_id}: ability/attack text set at {shrink:.0%} to fit")
    if "row" in groups:
        row_text = ink_near(scan, _cover(_drawn(_row, original, compose.WHITE), ("row_text",)), holes=False)
        _row(canvas, card, ink_colour(scan, row_text))
    canvas[..., 3] = scan[..., 3]
    return canvas, {"card": card, "edited": list(groups), "taken_off": mask}
