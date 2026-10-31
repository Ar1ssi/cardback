"""Edits made where a Scarlet & Violet / Mega Evolution card prints them: Energy that changes type, and damage.

An edit that changes only the Energy types of a card's abilities and attacks (its cost icons, and {X} in its effect
text: a type change's costs, "Follow the type") or an attack's damage keeps the print. Setting the block again
(modern.py) takes it all off and LaMa rebuilds what was under it: on a panel that is fine, on full art (Ultra Rare,
Special Illustration Rare) it wipes the art under the words. So instead:
- Cost icons: each printed icon of a type that changes is found in the cost column (its Mega Evolution sprite,
  meframe.icon, matched in grey) and covered with the new type's at the same place and size (compose.energy: the
  sprites share size and rim). The printed icon's rim reaches a little past the sprite's (about half a pixel): the
  old icon is filled from round it first, or its colour rings the new one.
- Energy in effect text: each printed symbol of a type that changes is found by matching our symbol's shape
  (compose.energy_symbol, at the print's text size) against the print's ink in a line of words (white print outlined
  over art: the fill's colour inside its keyline). It is filled from round it and the new type's symbol set in its
  place in the block's own ink (modern.print_ink: fill, keyline).
- Damage: every attack's printed damage is found by matching the figures as the composers set them
  (compose.damage_figures) on the print's ink, in the attacks' order down the card (a "30" also matches the end of a
  "130": the order tells them apart). A changed one is taken off (LaMa round it: on art the new figures need not cover
  the old ones) and the new one set at the old one's right edge and baseline, in the block's ink.
A print on which not all of it is found is set again as before.
"""
import itertools
import re

import cv2
import numpy as np

from . import compose, inpaint, layers, lifted, mecompose, meframe, typeset
from .imaging import dilate, fill_holes, luminance

W, H = 733, 1024
COST_X = (30, 300)                 # x0, x1: where a print's cost icons sit (the zone's rows: modern.MOVES_ZONE)
ICON_MATCH = 0.75                  # least match of a printed cost icon (normalized correlation inside its disc)
ICON_APART = 30                    # px: two cost icons' centres are at least this far apart
ICON_CLEAR = 1.5                   # px past the sprite's radius the printed icon is filled from round it
SYMBOL_GROW = 2                    # px round a printed symbol's shapes (and its keyline) taken off with it
SYMBOL_MATCH = 0.45                # least match of a printed symbol's shape with ours (outlined print over art
                                   # reads ~0.5)
SYMBOL_PAD = 3                     # px of the print round a symbol's shape matched with it
OUTLINE_NEAR = 70                  # RGB off the fill's / keyline's colour a pixel of outlined print lies within
OUTLINE_REACH = 3                  # px from its keyline a pixel of outlined print lies within
GLYPH_HOLE = 600                   # px: a keyline encloses at most this much of its glyph (bigger: the art)
WORD_SIDE, WORD_ROWS = (16, 56), 18  # px either side of a symbol / above its baseline where its line's words are
WORD_INK = 0.08                    # least mean ink there (a line of words)
FILL_RADIUS = 4                    # px: what is taken off is filled from this far round it (Telea)
DAMAGE = re.compile(r"\d+[+×-]?")  # a damage the composers set (as edit.DAMAGE, with its number)
DAMAGE_BASE = 600                  # baseline the damage figures are laid out on to be matched
DAMAGE_REACH = (24, 40)            # px the print's damage may sit off our right edge, and above / below our rows
DAMAGE_MATCH = 0.5                 # least match of a printed damage with ours
DAMAGE_PICKS = 4                   # best places kept per damage, the order chosen among them
DAMAGE_APART = 20                  # px: an attack's damage sits at least this far below the one before
LAMA_MARGIN = 48                   # px of the print round a hole LaMa sees
OFF, ON = "swap_off", "swap_on"    # layers kinds: the print taken off, set again (foil.build_modern's plates)
ICON = "swap_icon"                 # layers kind: the cost icons set (their cut-outs in the plates stay)
SYMBOL = re.compile(r"\{([A-Z])\}")


def _letters():
    from . import edit
    return {v: k for k, v in edit.LETTER.items()}


def _text_swap(old, new, table):
    """Whether effect text new is old with only its {X} symbols' types changed, each type into one; adds the
    changes to table {old letter: new letter}."""
    a, b = SYMBOL.split(old or ""), SYMBOL.split(new or "")
    if len(a) != len(b) or a[::2] != b[::2]:
        return False
    return all(table.setdefault(x, y) == y for x, y in zip(a[1::2], b[1::2]))


def _damage(a):
    return str(a.get("damage") or "").replace("x", "×")


def changes(original, card):
    """What an edit changes that can be made where the card prints it: {"costs": {old type: new type}, "text": {old
    type: new type}, "damage": {attack index: (old, new)}}, when card's abilities and attacks are original's with only
    Energy types (each type into one) and printed damages changed; else None."""
    a, b = original, card
    if len(a["abilities"]) != len(b["abilities"]) or len(a["attacks"]) != len(b["attacks"]):
        return None
    costs, text, damage = {}, {}, {}
    for x, y in zip(a["abilities"], b["abilities"]):
        if x.get("name") != y.get("name") or not _text_swap(x.get("effect"), y.get("effect"), text):
            return None
    for i, (x, y) in enumerate(zip(a["attacks"], b["attacks"])):
        if x.get("name") != y.get("name"):
            return None
        if _damage(x) != _damage(y):
            if not (DAMAGE.fullmatch(_damage(x)) and DAMAGE.fullmatch(_damage(y))):
                return None                         # a damage added or taken off: the block is set again
            damage[i] = (_damage(x), _damage(y))
        old, new = x.get("cost") or [], y.get("cost") or []
        if len(old) != len(new) or not all(costs.setdefault(o, n) == n for o, n in zip(old, new)):
            return None
        if not _text_swap(x.get("effect"), y.get("effect"), text):
            return None
    types = _letters()
    costs = {o: n for o, n in costs.items() if o != n}
    text = {types[o]: types[n] for o, n in text.items() if o != n}
    return {"costs": costs, "text": text, "damage": damage} if costs or text or damage else None


def _grey(a):
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def cost_r(card):
    return compose.COST_R if card["family"] == "ex" else mecompose.COST_R


def cost_icons(rgb, card, kind, count):
    """Centres of up to `count` printed cost icons of type kind, best matches first."""
    from . import modern
    x0, x1 = COST_X
    _, y0, _, y1 = modern.MOVES_ZONE[card["family"]]
    (crop, alpha, scx, scy, (ox, oy)), r = meframe.icon(kind, "cost")
    scale = cost_r(card) / r
    if abs(scale - 1) >= 0.03:                     # Dragon's is its name-bar icon, larger
        size = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
        crop, alpha = cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.resize(alpha, size)
        scx, scy, ox, oy = scx * scale, scy * scale, ox * scale, oy * scale
    score = cv2.matchTemplate(_grey(rgb[y0:y1, x0:x1]), _grey(crop), cv2.TM_CCOEFF_NORMED,
                              mask=(alpha > 0.5).astype(np.uint8))
    score = np.nan_to_num(score, nan=0, posinf=0, neginf=0)
    found = []
    while len(found) < count:
        _, best, _, (bx, by) = cv2.minMaxLoc(score)
        if best < ICON_MATCH:
            break
        found.append((x0 + bx + scx - ox, y0 + by + scy - oy))
        score[max(0, by - ICON_APART):by + ICON_APART, max(0, bx - ICON_APART):bx + ICON_APART] = 0
    return found


def outlined(rgba, box, fill, edge):
    """Coverage of print in a keyline over full art (Ultra Rare, Special Illustration Rare): pixels near the fill's
    colour with the keyline's colour within reach. The art round it is too busy to read print off a median."""
    c = rgba[..., :3]
    near_fill = np.abs(c - np.asarray(fill, np.float32)).max(-1) < OUTLINE_NEAR
    near_edge = np.abs(c - np.asarray(edge, np.float32)).max(-1) < OUTLINE_NEAR
    inside = fill_holes(near_edge, max_area=GLYPH_HOLE) & ~near_edge      # what a glyph's keyline encloses
    cov = near_fill & (dilate(near_edge, OUTLINE_REACH) | inside)
    out = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = cov[y0:y1, x0:x1]
    return out


def ink_cover(rgba, box, fill):
    """How far the print stands out from the panel in box, in units of its ink (lifted.standout), with the print's
    own sign (its fill dark or light): the sign standing out the more can be the cost icons'."""
    x0, y0, x1, y1 = box
    lum = luminance(np.clip(rgba[..., :3], 0, 255)).astype(np.float32)
    background = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), lifted.BACKGROUND).astype(np.float32)
    sign = 1 if luminance(np.asarray(fill, np.float32)[None])[0] < 128 else -1
    d = (background - lum)[y0:y1, x0:x1] * sign
    strokes = d > float(np.percentile(d, 99.5)) / 2
    ink = float(np.percentile(d[strokes], lifted.INK_SHARE)) if strokes.any() else 1.0
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = np.clip(d / max(ink, 1.0), 0, 1)
    return out


def _in_words(cov, gx, gy):
    """Whether print lies either side of a symbol at (gx, baseline gy): it sits in a line of words, not the art."""
    gx, gy = int(round(gx)), int(round(gy))
    rows = slice(gy - WORD_ROWS, gy)
    left = cov[rows, max(0, gx - WORD_SIDE[1]):max(0, gx - WORD_SIDE[0])]
    right = cov[rows, gx + WORD_SIDE[0]:gx + WORD_SIDE[1]]
    return min(left.mean() if left.size else 0, right.mean() if right.size else 0) > WORD_INK


def _template(shape, pad):
    """(the shape's crop with pad px round it, its top-left x, y)."""
    ys, xs = np.nonzero(shape > 0.02)
    y0, x0 = ys.min() - pad, xs.min() - pad
    return shape[y0:ys.max() + 1 + pad, x0:xs.max() + 1 + pad].astype(np.float32), x0, y0


def find_symbols(cov, box, kind, size, count):
    """[(score, gap centre x, baseline y)] of up to `count` printed symbols of type kind in effect text set at size:
    our symbol's shape matched against the print's ink, best first, each in a line of words."""
    ref_x, ref_y = 200.0, 200.0
    tmpl, x0, y0 = _template(compose.energy_symbol(kind, ref_x, ref_y, size), SYMBOL_PAD)
    zx0, zy0, zx1, zy1 = box
    score = cv2.matchTemplate(cov[zy0:zy1, zx0:zx1].astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
    score = np.nan_to_num(score, nan=0, posinf=0, neginf=0)
    found = []
    while len(found) < count:
        _, best, _, (bx, by) = cv2.minMaxLoc(score)
        if best < SYMBOL_MATCH:
            break
        score[max(0, by - tmpl.shape[0]):by + tmpl.shape[0], max(0, bx - tmpl.shape[1]):bx + tmpl.shape[1]] = 0
        gx, gy = zx0 + bx + ref_x - x0, zy0 + by + ref_y - y0
        if _in_words(cov, gx, gy):
            found.append((best, gx, gy))
    return found


def find_damages(cov, box, damages):
    """[(dx, dy)] per damage of damages (the attacks' printed damages, top to bottom): how far right and down of our
    figures on baseline DAMAGE_BASE the print has them; None when any is not found. Each is looked for in the damage
    column, and of each one's best places the order down the card that matches best is kept."""
    zx0, zy0, zx1, zy1 = box
    picks = []
    for text in damages:
        tmpl, x0, y0 = _template(compose.damage_figures(text, DAMAGE_BASE), SYMBOL_PAD)
        rx, ry = DAMAGE_REACH
        sx0, sx1 = max(zx0, x0 - rx), min(zx1, x0 + tmpl.shape[1] + rx)
        area = cov[zy0:zy1, sx0:sx1].astype(np.float32)
        if area.shape[0] < tmpl.shape[0] or area.shape[1] < tmpl.shape[1]:
            return None
        score = np.nan_to_num(cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED), nan=0, posinf=0, neginf=0)
        mine = []
        while len(mine) < DAMAGE_PICKS:
            _, best, _, (bx, by) = cv2.minMaxLoc(score)
            if best < DAMAGE_MATCH:
                break
            mine.append((best, sx0 + bx - x0, zy0 + by - y0))
            score[max(0, by - DAMAGE_APART):by + DAMAGE_APART, max(0, bx - tmpl.shape[1]):bx + tmpl.shape[1]] = 0
        if not mine:
            return None
        picks.append(mine)
    best = None
    for order in itertools.product(*picks):
        dys = [dy for _, _, dy in order]
        if all(b - a >= DAMAGE_APART for a, b in zip(dys, dys[1:])):
            total = sum(s for s, _, _ in order)
            if best is None or total > best[0]:
                best = (total, [(dx, dy) for _, dx, dy in order])
    return None if best is None else best[1]


def _block_ink(rgba, card):
    """The block's ink, (fill, keyline or None), and the print's ink coverage in its zone."""
    from . import modern
    zone = modern.MOVES_ZONE[card["family"]]
    ink = modern.print_ink(rgba, zone, modern.old_text(rgba, card))
    fill, edge = ink if ink else (modern._block_colour(rgba, card), None)
    cov = outlined(rgba, zone, fill, edge[0]) if edge else ink_cover(rgba, zone, fill)
    return fill, edge, cov


def find(rgba, original, card):
    """Where the print has what the edit changes, when it can be made there (changes): {"icons": [(old, new, cx,
    cy)], "symbols": [(old, new, gx, baseline, size)], "damage": [(old, new, dx, dy)]}; None when the edit changes
    more, or not all of it is found."""
    from . import edit, modern
    change = changes(original, card)
    if change is None:
        return None
    plan = {"icons": [], "symbols": [], "damage": []}
    for old, new in change["costs"].items():
        count = sum((a.get("cost") or []).count(old) for a in original["attacks"])
        found = cost_icons(rgba[..., :3], original, old, count)
        if len(found) < count:
            return None
        plan["icons"] += [(old, new, cx, cy) for cx, cy in found]
    zone = modern.MOVES_ZONE[original["family"]]
    cov = _block_ink(rgba, original)[2] if change["text"] or change["damage"] else None
    texts = [b.get("effect") or "" for b in original["abilities"] + original["attacks"]]
    for old, new in change["text"].items():
        count = sum(t.count(f"{{{edit.LETTER[old]}}}") for t in texts)
        # the print shrinks its text to fit as compose does: the size its symbols match best
        tries = [(find_symbols(cov, zone, old, compose.BODY_SIZE * k, count), compose.BODY_SIZE * k)
                 for k in compose.SHRINK_STEPS]
        found, size = max(tries, key=lambda t: (len(t[0]), sum(s for s, _, _ in t[0])))
        if len(found) < count:
            return None
        plan["symbols"] += [(old, new, gx, gy, size) for _, gx, gy in found]
    if change["damage"]:
        printed = [i for i, a in enumerate(original["attacks"]) if DAMAGE.fullmatch(_damage(a))]
        places = find_damages(cov, zone, [_damage(original["attacks"][i]) for i in printed])
        if places is None:
            return None
        for i, (dx, dy) in zip(printed, places):
            if i in change["damage"]:
                plan["damage"].append((*change["damage"][i], dx, dy))
    return plan


def _shifted(cover, dx, dy):
    return cv2.warpAffine(cover, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR)


def apply(rgba, plan, card):
    """The print with plan's icons, symbols and damages on their new values (card: the card as printed, for its
    layout). What is taken off and set again in the block's ink is recorded under the layers kinds OFF and ON."""
    out = rgba.copy()
    if plan["symbols"] or plan["damage"]:
        fill, edge, _ = _block_ink(rgba, card)
        grow = SYMBOL_GROW + (int(np.ceil(edge[1])) if edge else 0)
        symbols, damages = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        for old, _, gx, gy, size in plan["symbols"]:
            np.maximum(symbols, compose.energy_symbol(old, gx, gy, size), out=symbols)
        for old, _, dx, dy in plan["damage"]:
            np.maximum(damages, _shifted(compose.damage_figures(old, DAMAGE_BASE), dx, dy), out=damages)
        if symbols.any():
            out[..., :3] = _filled(out, dilate(symbols > 0.05, grow))
        if damages.any():                           # the new figures need not cover the old: LaMa, not a smear
            out = _lama(out, dilate(damages > 0.05, grow))
        with layers.kind(OFF):
            layers.record(np.maximum(symbols, damages))
        new = [compose.energy_symbol(t, gx, gy, size) for _, t, gx, gy, size in plan["symbols"]]
        new += [_shifted(compose.damage_figures(t, DAMAGE_BASE), dx, dy) for _, t, dx, dy in plan["damage"]]
        for cov in new:
            with layers.kind("swap_ink"):
                typeset.stamp(out, cov, fill, edge)
            with layers.kind(ON):
                layers.record(cov)
    r = cost_r(card)
    yy, xx = np.mgrid[:H, :W]
    for old, new, cx, cy in plan["icons"]:
        hole = np.hypot(xx - cx, yy - cy) <= r + ICON_CLEAR
        out[..., :3] = _filled(out, hole)
        with layers.kind(ICON):
            compose.energy(new, cx, cy, r, out)
    return out


def _filled(rgba, hole):
    """RGB with the hole filled from round it (what is set over it covers most of it)."""
    img = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    filled = cv2.inpaint(img, hole.astype(np.uint8), FILL_RADIUS, cv2.INPAINT_TELEA).astype(np.float32)
    return np.where(hole[..., None], filled, rgba[..., :3])


def _lama(rgba, hole):
    """rgba with the hole rebuilt by LaMa from the print round it (LAMA_MARGIN px of it)."""
    ys, xs = np.nonzero(hole)
    y0, y1 = max(0, ys.min() - LAMA_MARGIN), min(H, ys.max() + 1 + LAMA_MARGIN)
    x0, x1 = max(0, xs.min() - LAMA_MARGIN), min(W, xs.max() + 1 + LAMA_MARGIN)
    out = rgba.copy()
    out[y0:y1, x0:x1] = inpaint.inpaint(rgba[y0:y1, x0:x1], hole[y0:y1, x0:x1])
    return out
