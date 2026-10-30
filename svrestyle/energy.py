"""Energy cards in the Mega Evolution look.

Four frames, all cut from TCG Live scans:
- "basic": a Basic Energy of one of the eight types Mega Evolution prints, on that type's Basic Energy (mee-001..008).
  The frame is the whole card (the art is the type's own), so only the footer is set again: set code and number.
  Fairy, which Mega Evolution does not print, is Psychic's (mee-005) in Fairy's colours with its symbols (fairy.py).
- "gold": a Basic Energy on its type's gold Hyper Rare print instead (Paldea Evolved's Grass and Water, and the other
  six of sv01 to sv06.5: GOLD_TEMPLATES), the editor's choice; the footer set again the same way, the three gold
  stars after the number, and its foil the print's own (SvUltra, its plate and etch, the footer's print moved with
  it). Fairy's is Psychic's (sv03.5-207) made Fairy's: its art window and swoosh in Fairy's colours, the gold as it is.
- "regular": a Special Energy, the old art in the window of Team Rocket's Energy (me02.5-217), text on its panel.
- "full": a Special Energy whose print is a full art (Secret Rare and up): the art runs under the text, as on Ignition
  Energy (me02-124).
- "typed": a Special Energy that provides one type, on that type's Mega Evolution typed Special Energy (Telepathic
  Psychic Energy, me03-088, and the other seven: TYPED_TEMPLATES), the editor's choice: the print's own art (the
  type's symbol on a glass sphere) and Rare grade, the name's type printed as its symbol ("Charming {Y} Energy"), and
  the print's foil. Fairy's is Psychic's made Fairy's (fairy.typed_energy).
What a template prints (name, cost icons, text, footer) is painted out by LaMa and set again for this card; the old
art keeps its window, the old text on it painted out. Footers and text layout are the Trainers' (trainer.py), the
icons the Mega Evolution ones. SV / Mega Evolution Energy is already in this look: frame "print", the card as TCG
Live prints it (its own scan and foil), with nothing to edit; it is there to be listed and put in a set.
"""
import re
import warnings

import cv2
import numpy as np

from . import cache, compose, foil, frame, inpaint, layers, mecompose, modern, tcg, trainer
from . import typeset as ts
from .imaging import dilate, load_rgba, luminance, save_rgba

W, H = 733, 1024
VERSION = 9
FULL_RARITY = ("secret", "ultra", "illustration", "hyper", "special")
SPECIAL_TEMPLATES = {"regular": "me02.5-217", "full": "me02-124"}
BASIC_TEMPLATES = {"Grass": "mee-001", "Fire": "mee-002", "Water": "mee-003", "Lightning": "mee-004",
                   "Psychic": "mee-005", "Fighting": "mee-006", "Darkness": "mee-007", "Metal": "mee-008",
                   "Fairy": "mee-005"}     # Fairy: Psychic's made Fairy's (fairy.basic_energy); its foil is Psychic's
GOLD_TEMPLATES = {"Grass": "sv02-278", "Fire": "sv03-230", "Water": "sv02-279", "Lightning": "sv01-257",
                  "Psychic": "sv03.5-207", "Fighting": "sv01-258", "Darkness": "sv06.5-098", "Metal": "sv06.5-099",
                  "Fairy": "sv03.5-207"}   # Fairy: Psychic's made Fairy's (fairy.gold_energy)
GOLD_RARITY = "Hyper rare"
TYPED_TEMPLATES = {"Grass": "me03-086", "Fire": "me04-086", "Water": "me04-084", "Lightning": "me05-084",
                   "Psychic": "me03-088", "Fighting": "me03-087", "Darkness": "me05-083", "Metal": "me04-085",
                   "Fairy": "me03-088"}  # Fairy: Psychic's made Fairy's (fairy.typed_energy)
FRAMES = ("regular", "full", "typed")           # what a Special Energy can be set on; "basic" and "gold" are for Basic
                                                # Energy, "print" for Energy that is in the look already
BASIC_FRAMES = ("basic", "gold")
BASIC_NAME = re.compile(rf"(?:Basic )?({'|'.join(BASIC_TEMPLATES)}) Energy")
# Art windows (x0, y0, x1, y1): a full-art print's, and each template's. A print with a text panel has its window found
# on its scan (old_window).
OLD_ART = (31, 141, 702, 947)
WINDOW = {"regular": (29, 140, 703, 655), "full": (29, 140, 703, 948)}
# Zones the template prints on (cleared before the card's own are set): name, cost icons, footer, panel text.
# The footer zones stop short of the bar's rounded ends, which print nothing: Ignition's icons run from x 585 to 697.
FOOT_LEFT, FOOT_RIGHT = (38, 950, 340, 998), (578, 944, 700, 1000)
# The footer bar runs level: in its rows, what was cleared is set again from the bar's own rows where nothing prints
# (LaMa leaves ghosts of the regulation mark and icons in so long a hole).
FOOT_BAR, FOOT_PLAIN = (949, 997), (350, 560)
CLEAR = {
    "regular": [(40, 70, 545, 138), (565, 66, 706, 138), FOOT_LEFT, FOOT_RIGHT, (50, 685, 690, 925)],
    "typed": [(40, 70, 545, 138), (565, 66, 706, 138), FOOT_LEFT, FOOT_RIGHT, (50, 685, 690, 925)],   # same frame
    "full": [(40, 70, 515, 138), (520, 66, 706, 138), FOOT_LEFT, FOOT_RIGHT],
    "basic": [FOOT_LEFT],
    "gold": [(38, 952, 250, 998)],               # the gold print's set box, number and stars, set again for this card
}
# The gold print's Hyper Rare stars (x0, y0, x1, y1), the same on each type's: set again STAR_GAP px after the number
# (the prints' own gap), lifted off the footer's gold where they stand out from it by STAR_INK levels (soft below).
GOLD_STARS = (200, 956, 244, 994)
STAR_GAP, STAR_INK = 8.5, (12, 40)
GOLD_LINES_GROW = 2      # px round a gold print's footer print its etch's lines are not read from
TITLE_REACH, TITLE_INK = 3, 90   # the title line's symbol: px past its radius, mean RGB under which a pixel is its ink
LINES_PERIOD, LINES_AROUND, LINES_SHIFTS = (4, 24), 20, 12   # the etch's lines: period range, px read round a hole,
                                                             # periods a hole's pixel may be taken from
# The name bar's icons and the footer's, the last one's centre (measured on me02.5-217 and the typed prints; the footer's
# are the attack cost icons at their own size, three of them from x 585 to 697 as on Ignition Energy)
COST_X, COST_Y, COST_R, COST_STEP = 669.5, 104.5, 27.5, 55.5
FOOT_COST_X, FOOT_COST_Y, FOOT_COST_R, FOOT_STEP = 678.5, 972.5, 18.5, 38.0
MAX_COST = 3                                                      # icons the bar holds (Ignition Energy's three)
# The name as the Mega Evolution Special Energy prints set it (ten prints, me02-124 to me05-084): 2 px right of and below
# the Trainers' (trainer.NAME_X, NAME_BASE). The typed prints set their type as its symbol between the words, NAME_GAPS
# px clear of ink before and after it whatever its width (at trainer.NAME_XSCALE; NAME_SLOT: the symbol and its gaps,
# about, to fit a name), the symbol lifted off the print's name (NAME_ROWS; ink is darker than NAME_INK, NAME_GAP px or
# more apart from the letters, NAME_DARK when fully inked).
NAME_X, NAME_BASE = trainer.NAME_X + 2, trainer.NAME_BASE + 2
NAME_GAPS, NAME_SLOT = (15.0, 17.5), 75.0
NAME_ROWS, NAME_COLS, NAME_INK, NAME_GAP, NAME_DARK = (78, 136), (40, 560), 110, 9, 35
# The pixel just outside the window is a blend of the art's edge and the frame beyond it, art this much (measured on
# me02.5-217 and me02-124; a full art's bottom meets the footer's own highlight, which is kept). The template's seam
# holds its own art, so it is made again from this card's: left as it is, it rings the art in another colour and the
# art looks laid on the frame.
SEAM = {"regular": {"left": 0.65, "right": 0.6, "top": 0.35, "bottom": 0.25},
        "full": {"left": 0.65, "right": 0.6, "top": 0.3}}
RIM = 3                                                          # px of the old art's edge left off
TEXT_LAST_BASE = 905                                              # full art: the last line's baseline
# The text as Special Energy prints it (me02.5-217, me02-124): tighter than the Trainers'; across a paragraph break the
# baseline steps PARA_GAP px, not a line and a gap.
TEXT_PITCH, PARA_GAP, PANEL_CENTRE, PANEL_ROOM = 29.0, 50.0, 800.0, 190.0
# Where an old print's window sits (sm1-136, sm3.5-69, sm4-100, swsh1-186, xy4-111): its edges are found within these.
WINDOW_SEARCH = {"left": (22, 38), "right": (696, 710), "top": (128, 152), "bottom": (640, 690)}
WINDOW_SIZE = ((655, 685), (470, 560))                            # width and height range of those windows
EDGE = 35                                                       # luminance step that makes an edge
NAME_RIGHT = 528
OLD_TEXT = (40, 760, 600, 830)                                    # the old print's text on its art (full-art prints)
# Black & White and XY prints set a "Special Energy" pill over the art's top left (xy6-97, xy4-111, bw4-92): found by
# its grey foot (rows the art is colour in on other prints) in PILL_PROBE, painted out over PILL.
PILL, PILL_PROBE = (31, 128, 348, 150), (60, 139, 330, 144)
PILL_GREY = 0.95                                                   # share of PILL_PROBE that is unsaturated
# Old Special Energy wording -> what the new cards print.
PROVIDES = "As long as this card is attached to a Pokémon, it provides "
REWORD = [
    (r"^Attach [^.]+? to 1 of your Pokémon\.\s*", ""),
    (r"\s*\(Has no effect other than providing Energy\.\)", ""),
    (r"^While in play, (?:[A-Z][\w’' ]+? Energy|this card) provides ", PROVIDES),
    (r"^While in play, ", "As long as this card is attached to a Pokémon, "),
]
PROVIDED = re.compile(r"provides ((?:\{[A-Z]\})+) Energy")


def is_full(card):
    return any(word in (card.get("rarity_name") or "").lower() for word in FULL_RARITY)


def reworded(paragraphs, name):
    """The old wording of what the card provides, as the new cards print it."""
    out = []
    for para in paragraphs:
        para = re.sub(rf"^{re.escape(name)} provides ", PROVIDES, para)
        for old, new in REWORD:
            para = re.sub(old, new, para)
        out.append(para.strip())
    return [p for p in out if p]


def provided(effect):
    """The types the first 'provides {C}{C} Energy' names, for the name bar's icons."""
    m = PROVIDED.search(effect or "")
    return [ts.ENERGY_CODES[c] for c in re.findall(r"\{([A-Z])\}", m[1])][:MAX_COST] if m else []


def read_card(card_id):
    c = tcg.card(card_id)
    name = c.get("name", "")
    if c.get("category") != "Energy":
        raise SystemExit(f"{card_id} is {name!r} ({c.get('category')}): not an Energy card")
    set_id = c["set"]["id"]
    s = tcg.set_info(set_id)
    if (s.get("serie") or {}).get("id") == trainer.POCKET_SERIES:
        raise SystemExit(f"{card_id} is {name!r}, a Pokemon TCG Pocket card: there is no printed card to restyle")
    in_look = modern.is_modern(card_id)
    rarity = c.get("rarity") or ""
    official = (s.get("cardCount") or {}).get("official")
    card = {
        "id": card_id, "prefix": "", "name": name, "illustrator": c.get("illustrator") or "",
        "rarity_name": rarity, "rarity": mecompose.rarity_symbol(rarity), "reg_mark": c.get("regulationMark") or "",
        "set_code": mecompose.set_code(set_id, s), "year": (s.get("releaseDate") or "")[:4],
        "number": mecompose.card_number(c.get("localId", ""), official, s),
    }
    raw = c.get("effect") or ""
    basic = BASIC_NAME.fullmatch(name)
    if basic and c.get("energyType") != "Special":
        card.update(energy="basic", frame="print" if in_look else "basic", type=basic[1], effect="", cost=[])
        return card
    if re.fullmatch(r"(?:Basic )?\w+ Energy", name) and not raw.strip():
        raise SystemExit(f"{card_id} is {name!r}: the Mega Evolution Basic Energy comes in these types: "
                         f"{', '.join(BASIC_TEMPLATES)}")
    if not raw.strip():
        raise SystemExit(f"{card_id} is {name!r}: TCGdex gives no text for this Special Energy")
    paragraphs = reworded(trainer.effect_paragraphs(raw), name)
    card.update(energy="special", effect="\n".join(paragraphs), cost=provided(" ".join(paragraphs)))
    if not card["cost"] and not in_look:
        warnings.warn(f"{card_id}: no Energy it provides could be read from its text: the name bar prints no icons "
                      "(set them in the editor)")
    card["frame"] = "print" if in_look else "full" if is_full(card) else "regular"
    return card


# ---------------------------------------------------------------- frame and art
def typed_type(card):
    """The type a typed Special Energy shows (the one type it provides, when there is a typed print of it), or None."""
    provided = set(card.get("cost") or [])
    return next(iter(provided)) if len(provided) == 1 and provided <= set(TYPED_TEMPLATES) else None


def template_id(card):
    if card["energy"] == "basic":
        return (GOLD_TEMPLATES if card["frame"] == "gold" else BASIC_TEMPLATES)[card["type"]]
    if card["frame"] == "typed":
        if typed_type(card) is None:
            raise ValueError(f"{card['id']}: the typed frame shows one type: the Energy it provides must be one of "
                             f"{', '.join(TYPED_TEMPLATES)}")
        return TYPED_TEMPLATES[typed_type(card)]
    return SPECIAL_TEMPLATES[card["frame"]]


def _typed_template(type_name):
    """A type's typed Special Energy print, Fairy's made from Psychic's (fairy.typed_energy)."""
    if type_name == "Fairy":
        from . import fairy
        return fairy.typed_energy()
    return tcg.scan(TYPED_TEMPLATES[type_name])


def _template(card):
    """The template's scan, a derived type's made from its base's (Fairy: fairy.basic_energy / gold_energy /
    typed_energy)."""
    if card["frame"] == "typed":
        template_id(card)                       # a card with no typed print is refused here
        return _typed_template(typed_type(card))
    template = tcg.scan(template_id(card))
    if card["energy"] == "basic" and card["type"] == "Fairy":
        from . import fairy
        template = fairy.gold_energy(template) if card["frame"] == "gold" else fairy.basic_energy(template)
    return template


def _clear(card):
    """The template with its name, icons, text and footer painted out; cached."""
    frame_name = card["frame"]
    from . import fairy
    if frame_name in BASIC_FRAMES:
        name = f"{'gold-' if frame_name == 'gold' else ''}{card['type']}{fairy.tag(card['type'])}"
    elif frame_name == "typed":
        template_id(card)
        name = f"typed-{typed_type(card)}{fairy.tag(typed_type(card))}"
    else:
        name = frame_name
    out = cache.path("assets", f"energy-v{VERSION}-{name}.png")
    if out.exists():
        return load_rgba(out)
    template = _template(card)
    mask = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in CLEAR[frame_name]:
        mask[y0:y1, x0:x1] = True
    result = inpaint.inpaint(template, mask)
    if frame_name != "basic":
        y0, y1 = FOOT_BAR
        bar = np.median(template[y0:y1, slice(*FOOT_PLAIN), :3], axis=1)
        level = mask[y0:y1]
        result[y0:y1, :, :3][level] = np.broadcast_to(bar[:, None], (y1 - y0, W, 3))[level]
    save_rgba(result, out)
    return result


def _edge(line, lo, hi, last=False):
    steps = np.flatnonzero(np.abs(np.diff(line[lo:hi + 1])) > EDGE)
    return None if not steps.size else lo + int(steps[-1 if last else 0]) + 1


def old_window(card):
    """(x0, y0, x1, y1) of the art on a print with a text panel, found from the frame's edges on its scan.
    A full-art print's is OLD_ART."""
    if card["frame"] == "full":
        return OLD_ART
    grey = tcg.scan(card["id"])[..., :3].mean(axis=2)
    found = {}
    for side, (lo, hi) in WINDOW_SEARCH.items():
        across = (280, 330, 380, 430, 480) if side in ("left", "right") else (150, 250, 350, 450, 550)
        hits = [_edge(grey[y] if side in ("left", "right") else grey[:, y], lo, hi, last=side == "right")
                for y in across]
        hits = [h for h in hits if h is not None]
        if not hits:
            raise SystemExit(f"{card['id']}: no art window found on its scan ({side} edge): the frame has no known "
                             "layout")
        found[side] = int(np.median(hits))
    x0, y0, x1, y1 = found["left"], found["top"], found["right"], found["bottom"]
    if not (WINDOW_SIZE[0][0] <= x1 - x0 <= WINDOW_SIZE[0][1] and WINDOW_SIZE[1][0] <= y1 - y0 <= WINDOW_SIZE[1][1]):
        raise SystemExit(f"{card['id']}: the art window found on its scan ({x1 - x0}x{y1 - y0}) is not one of the "
                         "known Special Energy frames (Sun & Moon, Sword & Shield, XY, Black & White)")
    return x0, y0, x1, y1


def old_print(card, scan):
    """bool HxW: what the old print sets on its art (a full art's text, a Black & White or XY print's pill)."""
    mask = np.zeros((H, W), bool)
    if card["frame"] == "full":
        x0, y0, x1, y1 = OLD_TEXT
        mask[y0:y1, x0:x1] = True
        return mask
    x0, y0, x1, y1 = PILL_PROBE
    probe = scan[y0:y1, x0:x1, :3]
    if (np.ptp(probe, axis=2) < 25).mean() >= PILL_GREY:
        x0, y0, x1, y1 = PILL
        mask[y0:y1, x0:x1] = True
    return mask


def old_art(card):
    """The old print's art, what it sets on it painted out, scaled into the template's window; cached."""
    frame_name = card["frame"]
    out = cache.path("art", f"energy-v{VERSION}-{frame_name}-{card['id']}.png")
    if out.exists():
        return load_rgba(out)
    scan = tcg.scan(card["id"])
    ox0, oy0, ox1, oy1 = old_window(card)
    wx0, wy0, wx1, wy1 = WINDOW[frame_name]
    inside = np.s_[oy0 + RIM:oy1 - RIM, ox0 + RIM:ox1 - RIM]     # the old print's rim and its shading are left off
    inset, mask = scan[inside], old_print(card, scan)[inside]
    if mask.any():                      # painted out on the art alone: LaMa would carry the old frame into the hole
        inset = inpaint.inpaint(inset, mask)
    art = cv2.resize(inset, (wx1 - wx0, wy1 - wy0), interpolation=cv2.INTER_CUBIC)
    result = np.zeros((H, W, 4), np.float32)
    result[wy0:wy1, wx0:wx1] = art
    result[..., 3] = 255
    save_rgba(result, out)
    return result


def set_art(canvas, art, frame_name):
    """The art in the template's window, the seam round it blended from the art's edge and the frame beyond."""
    x0, y0, x1, y1 = WINDOW[frame_name]
    canvas[y0:y1, x0:x1, :3] = art[y0:y1, x0:x1, :3]
    rows, cols = slice(y0, y1), slice(x0, x1)
    sides = {"left": ((rows, x0 - 1), (rows, x0 - 2), (rows, x0)),          # (seam, frame beyond it, art's edge)
             "right": ((rows, x1), (rows, x1 + 1), (rows, x1 - 1)),
             "top": ((y0 - 1, cols), (y0 - 2, cols), (y0, cols)),
             "bottom": ((y1, cols), (y1 + 1, cols), (y1 - 1, cols))}
    for side, a in SEAM[frame_name].items():
        seam, beyond, edge = sides[side]
        canvas[(*seam, slice(0, 3))] = a * canvas[(*edge, slice(0, 3))] + (1 - a) * canvas[(*beyond, slice(0, 3))]


# ---------------------------------------------------------------- drawing
def _fit(card, width, slot=0.0):
    """(size, xscale) the name is set at: the prints' squeeze, more for a long name, then a smaller size. width: the
    name's text at full size and no squeeze; slot: a symbol's gap (squeezed with the text)."""
    size, xscale = trainer.NAME_SIZE, trainer.NAME_XSCALE
    xscale = min(xscale, (NAME_RIGHT - NAME_X) / max(width + slot / trainer.NAME_XSCALE, 1))
    if xscale < trainer.NAME_MIN_XSCALE:
        size, xscale = size * xscale / trainer.NAME_MIN_XSCALE, trainer.NAME_MIN_XSCALE
        warnings.warn(f"name {card['name']!r} set at {size:.0f} px to fit the bar")
    return size, xscale


def name(canvas, card):
    size, xscale = _fit(card, ts.text_width(card["name"], ts.BOLD, trainer.NAME_SIZE))
    with layers.kind("name"):
        ts.stamp(canvas, ts.coverage([dict(text=card["name"], face=ts.BOLD, size=size, x=NAME_X, base=NAME_BASE,
                                           xscale=xscale)]),
                 np.array(frame.DARK_INK, np.float32))
    return xscale


TYPE_WORD = re.compile(rf"(.*\S)\s+({'|'.join(TYPED_TEMPLATES)})\s+(Energy)")


def name_glyph(type_name):
    """A type's symbol as its typed print sets it in the name ("Telepathic {P} Energy"; Fairy's: the Fairy print's,
    fairy.typed_energy): the ink between the name's words lifted off the print, as coverage of the name line's dark
    against its bar, NAME_ROWS tall (its rows as the print sets them on its baseline). float32"""
    lum = luminance(_typed_template(type_name)[..., :3])
    y0, y1 = NAME_ROWS
    x0, x1 = NAME_COLS
    band = lum[y0:y1, x0:x1]
    ink = band < NAME_INK
    cols = np.flatnonzero(ink.any(axis=0))
    groups = [[cols[0], cols[0]]]
    for x in cols[1:]:
        if x - groups[-1][1] > NAME_GAP:
            groups.append([x, x])
        else:
            groups[-1][1] = x
    if len(groups) < 3:
        raise ValueError(f"{TYPED_TEMPLATES[type_name]}: no symbol found in its name")
    left, right = groups[-2][0] - 2, groups[-2][1] + 3            # the symbol: between the word and "Energy"
    bar = np.array([np.median(row[~dark]) for row, dark in zip(band, ink)], np.float32)[:, None]
    return np.clip((bar - band[:, left:right]) / (bar - NAME_DARK), 0, 1).astype(np.float32)


def typed_name(canvas, card):
    """A typed Special Energy's name as the prints set it: the type before "Energy" printed as its symbol
    ("Charming Fairy Energy" -> "Charming {Y} Energy"), NAME_GAPS clear of the words. A name without a type before
    "Energy" is set as it is. -> the squeeze it was set at"""
    m = TYPE_WORD.fullmatch(card["name"])
    if not m:
        return name(canvas, card)
    word, last = m[1], m[3]
    size, xscale = _fit(card, sum(ts.text_width(t, ts.BOLD, trainer.NAME_SIZE) for t in (word, last)), NAME_SLOT)
    k = size / trainer.NAME_SIZE
    sx, sy = k * xscale / trainer.NAME_XSCALE, k          # the gaps and symbol, squeezed and shrunk with the text

    def text(t, x):
        return ts.coverage([dict(text=t, face=ts.BOLD, size=size, x=x, base=NAME_BASE, xscale=xscale)])

    def inked(cov):
        return np.flatnonzero(cov.max(axis=0) > 0.5)
    first = text(word, NAME_X)
    cover = name_glyph(m[2])
    cols = inked(cover)
    at = inked(first)[-1] + 1 + NAME_GAPS[0] * sx               # the symbol's first inked column
    past = at + (cols[-1] + 1 - cols[0] + NAME_GAPS[1]) * sx
    second = text(last, past - inked(text(last, 0))[0])
    y0 = NAME_ROWS[0]
    glyph = np.zeros((H, W), np.float32)
    glyph[y0:y0 + cover.shape[0]] = cv2.warpAffine(
        cover, np.float32([[sx, 0, at - sx * cols[0]], [0, sy, (NAME_BASE - y0) * (1 - sy)]]),
        (W, cover.shape[0]), flags=cv2.INTER_LINEAR)
    with layers.kind("name"):
        ts.stamp(canvas, np.maximum(np.maximum(first, second), glyph), np.array(frame.DARK_INK, np.float32))
    return xscale


def cost_icons(canvas, card):
    for i, t in enumerate(card["cost"][::-1]):
        with layers.kind("type_icon"):
            compose.energy(t, COST_X - i * COST_STEP, COST_Y, COST_R, canvas, kind="name")
        with layers.kind("icon"):               # lit on the plate, as the print's own
            compose.energy(t, FOOT_COST_X - i * FOOT_STEP, FOOT_COST_Y, FOOT_COST_R, canvas, kind="cost")


def text(canvas, card):
    full = card["frame"] == "full"
    paragraphs = [p for p in card["effect"].split("\n") if p.strip()]
    for k in trainer.SHRINK_STEPS:
        lines, rows = trainer.lay_paragraphs(paragraphs, k, tracking=(trainer.BODY_TRACK, trainer.BODY_MIN_TRACK))
        pitch, gap = TEXT_PITCH * k, PARA_GAP * k
        ys = [idx * pitch + (row - idx) * (gap - pitch) for idx, (row, *_) in enumerate(lines)]
        span = ys[-1] if ys else 0
        if span <= PANEL_ROOM or full:
            break
    else:
        warnings.warn(f"{card['id']}: the text runs taller than the panel holds")
    first = TEXT_LAST_BASE - span if full else PANEL_CENTRE - span / 2
    ink, edge = (compose.WHITE, compose.ART_EDGE) if full else (np.array(frame.DARK_INK, np.float32), None)
    size = trainer.BODY_SIZE * k
    items, drawn = [], np.zeros((H, W), np.float32)
    for y, (row, runs, tracking, types, indent, bullet) in zip(ys, lines):
        base, marks = first + y, []
        items += ts.line_items(runs, tracking, size, trainer.BODY_X + indent, base, marks)
        if bullet:
            items.append(dict(text=trainer.BULLET, face=trainer.BODY_FACE, size=size, x=trainer.BODY_X, base=base))
        for t, cx in zip(types, marks):
            np.maximum(drawn, compose.energy_symbol(t, cx, base, size), out=drawn)
    with layers.kind("text"):
        ts.stamp(canvas, np.maximum(ts.coverage(items), drawn), ink, edge)


def footer(canvas, card):
    """The Energy footer: regulation mark and set boxes, the number, a Special Energy's rarity mark (no illustrator
    line; the template keeps its copyright). -> the number's right edge (x)"""
    full = card["frame"] == "full"
    ink = compose.WHITE if full else np.array(frame.DARK_INK, np.float32)
    keyline = compose.LIGHT_KEYLINE if full else None
    dark = np.array([38, 33, 38], np.float32)
    with layers.kind("footer_box"):
        if card["reg_mark"]:
            ts.stamp(canvas, ts.rounded_box(44.5, 960.5, 62.5, 987.5, 3), dark)
            ts.stamp(canvas, ts.rounded_box(45.6, 961.6, 61.4, 986.4, 2.2), compose.WHITE)
            ts.stamp(canvas, ts.coverage([dict(text=card["reg_mark"], face=ts.COND_BOLD, size=19.5, x=53.6,
                                               base=981, anchor="c")]), dark)
        if card["set_code"]:
            code_w, en_w, box_x1 = compose._set_box(card["set_code"])
            ts.stamp(canvas, ts.rounded_box(67.5, 960.5, box_x1, 987.5, 3.2), dark)
            ts.stamp(canvas, ts.rounded_box(69.0, 962.0, box_x1 - 1.5, 986.0, 2.4), compose.WHITE)
            ts.stamp(canvas, ts.rounded_box(70.2, 963.2, box_x1 - 2.7, 984.8, 1.8), [28, 24, 28])
            left = (67.5 + box_x1) / 2 - (code_w + 1.5 + en_w) / 2
            ts.stamp(canvas, ts.coverage([dict(text=card["set_code"], face=ts.COND_BOLD, size=19.5, x=left,
                                               base=981, xscale=0.8),
                                          dict(text="EN", face=ts.COND_BOLD, size=11.5, x=left + code_w + 1.5,
                                               base=981, xscale=0.85)]), compose.WHITE)
            num_x = box_x1 + 8.5
        else:
            num_x = 71.0 if card["reg_mark"] else 46.0
    style = compose.FOOTER["number"]
    right = num_x + ts.text_width(card["number"], ts.HEAVY_ITALIC, style["size"], style["xscale"])
    with layers.kind("footer"):
        ts.stamp(canvas, ts.coverage([dict(text=card["number"], face=ts.HEAVY_ITALIC, size=style["size"], x=num_x,
                                           base=style["base"], xscale=style["xscale"])]), ink, keyline)
        if card["energy"] == "special":
            ts.stamp(canvas, compose.rarity_mark(card["rarity"], right + 6, 980.5), ink, keyline)
    return right


def gold_stars(card):
    """The gold print's Hyper Rare stars, lifted off its footer: (RGB, coverage, x0, y0) of GOLD_STARS."""
    template = _template(card)[..., :3]
    x0, y0, x1, y1 = GOLD_STARS
    crop = template[y0:y1, x0:x1]
    plain = np.median(template[y0:y1, slice(*FOOT_PLAIN)], axis=1)[:, None]      # the footer's gold, row by row
    lo, hi = STAR_INK
    cover = np.clip((np.abs(crop - plain).max(-1) - lo) / (hi - lo), 0, 1)
    return crop, cover, x0, y0


def set_stars(canvas, card, right):
    """The gold print's stars STAR_GAP px after the number (right: its right edge)."""
    crop, cover, x0, y0 = gold_stars(card)
    first = np.flatnonzero(cover.max(axis=0) > 0.5)[0]
    dx = int(round(right + STAR_GAP - (x0 + first)))
    h, w = cover.shape
    region = canvas[y0:y0 + h, x0 + dx:x0 + dx + w, :3]
    region[:] = region * (1 - cover[..., None]) + crop * cover[..., None]
    placed = np.zeros((H, W), np.float32)
    placed[y0:y0 + h, x0 + dx:x0 + dx + w] = cover
    with layers.kind("footer_stars"):
        layers.record(placed)
    return dx


def compose_card(card_id, card=None):
    """-> (RGBA float32 733x1024, report dict). card: read_card's dict, possibly edited (default: read it)."""
    card = card or read_card(card_id)
    if card["frame"] == "print":
        return tcg.scan(card_id).copy(), {"card": card}
    canvas = _clear(card).copy()
    if card["energy"] == "basic":
        right = footer(canvas, card)
        if card["frame"] == "gold":
            set_stars(canvas, card, right)
        return canvas, {"card": card}
    if card["frame"] == "typed":                # the print's own art, its type in the name as its symbol
        xscale = typed_name(canvas, card)
    else:
        set_art(canvas, old_art(card), card["frame"])
        xscale = name(canvas, card)
    cost_icons(canvas, card)
    text(canvas, card)
    footer(canvas, card)
    return canvas, {"card": card, "name_xscale": round(xscale, 3)}


# ---------------------------------------------------------------- holofoil
FOIL_CUT = ("type_icon", "text", "footer", "footer_box")
FOIL_FRAME = SPECIAL_TEMPLATES["full"]


def _old_art_plate(card, record):
    """The old print's plate over its art, in the template's window, its text filled in."""
    old = foil.plate(record)
    frame_name = card["frame"]
    if frame_name == "full":
        x0, y0, x1, y1 = OLD_TEXT
        marks = np.zeros((H, W), bool)
        marks[y0:y1, x0:x1] = old[y0:y1, x0:x1] < 60
        old = foil.fill(old, dilate(marks, 4))
    ox0, oy0, ox1, oy1 = old_window(card)
    wx0, wy0, wx1, wy1 = WINDOW[frame_name]
    inside = np.s_[oy0 + RIM:oy1 - RIM, ox0 + RIM:ox1 - RIM]
    inset = old[inside]
    if frame_name != "full":
        pill = old_print(card, tcg.scan(card["id"]))[inside]
        if pill.any():
            inset = foil.fill(inset, pill)
    return cv2.resize(inset, (wx1 - wx0, wy1 - wy0), interpolation=cv2.INTER_CUBIC)


def _fill_footer(img):
    zone = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in (FOOT_LEFT, FOOT_RIGHT):
        zone[y0:y1, x0:x1] = True
    return foil.fill(img, zone)


def build_foil(card, canvas, drawn):
    """Same shape as foil.build(): what TCG Live's foil of the old print becomes on the new frame.
    full-art Special Energy: the old print's foil on the art, Ignition Energy's round it.
    regular Special Energy: SvHolo on the art when the old print was foil, and the template's reverse holo (the
    header lettering and text panel) when it had one.
    Basic Energy: the template's reverse holo when the old print had one.
    Nothing for a print without foil."""
    if card["frame"] == "print":
        return _print_foil(card, canvas)
    if card["frame"] == "gold":
        return _gold_foil(card, canvas, drawn)
    if card["frame"] == "typed":
        return _print_foil(card, canvas, template_id(card))
    out = {"front": foil.front_texture(canvas), "base": None, "reverse": None}
    try:
        old = foil.printings(card["id"])
    except ValueError:
        warnings.warn(f"{card['id']}: TCG Live has no foil for this print: the card is set without")
        return out
    alpha = canvas[..., 3] / 255
    special, frame_name = card["energy"] == "special", card["frame"]
    if special and foil.is_foil(old["base"]):
        window = WINDOW[frame_name]
        x0, y0, x1, y1 = window
        if frame_name == "full":
            img = foil.plate(foil.printings(FOIL_FRAME)["base"])
            img[y0:y1, x0:x1] = _old_art_plate(card, old["base"])
            img = foil.cut(_fill_footer(img), drawn, FOIL_CUT) * alpha
            record = old["base"]
            spec = (foil.legacy_shader(record.get("_s") or ""), foil.material_name(record.get("_f") or ""),
                    record.get("_cc") or None)
        else:
            img = np.zeros((H, W), np.float32)
            img[y0:y1, x0:x1] = _old_art_plate(card, old["base"])
            img, spec = img * alpha, foil.ME_HOLO
        out["base"] = (spec, foil.plate_texture(img), {})
    if frame_name != "full" and old["reverse"]:
        template = foil.printings(template_id(card))["reverse"]
        if template:
            img = foil.plate(template)
            if special:
                x0, y0, x1, y1 = CLEAR["regular"][4]
                panel = np.zeros((H, W), bool)
                panel[y0:y1, x0:x1] = True
                img = foil.cut(foil.fill(img, panel & (img < 60)), drawn, ("text",))
            out["reverse"] = (foil.ME_REVERSE, foil.plate_texture(img * alpha), {})
    return out


def _etch(record):
    """A printing's etch in card space (luminance, float32 0..255), or None."""
    tex = foil.texture(record, record.get("_e"))
    if tex is None or tex.shape[0] != foil.TEX:
        return None
    lum = tex[..., :3].mean(-1) if tex.ndim == 3 else tex
    return lum[:, tcg.BLEED_X:tcg.BLEED_X + W].astype(np.float32)


def _footer_print(scan):
    """What a gold print prints on its footer's gold (set box, number, stars), round it GOLD_LINES_GROW px. bool HxW"""
    c = scan[..., :3]
    x0, y0, x1, y1 = CLEAR["gold"][0]
    plain = np.median(c[y0:y1, slice(*FOOT_PLAIN)], axis=1)[:, None]
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = np.abs(c[y0:y1, x0:x1] - plain).max(-1) > STAR_INK[0]
    return dilate(out, GOLD_LINES_GROW)


def _gold_lines():
    """The etched lines under a gold print's footer, its print taken off: per pixel, the median of the gold prints
    whose footer prints nothing there (their etches share the pattern there). Cached. float32 HxW"""
    out = cache.path("assets", f"energy-v{VERSION}-gold-lines.png")
    if out.exists():
        return np.asarray(cv2.imread(str(out), cv2.IMREAD_GRAYSCALE), np.float32)
    stack = []
    for card_id in sorted(set(GOLD_TEMPLATES.values())):
        record = foil.printings(card_id)["base"]
        etch = _etch(record)
        if etch is not None:
            stack.append(np.where(_footer_print(tcg.scan(card_id)), np.nan, etch))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pixels every print prints on: nan, filled below
        lines = np.nanmedian(np.stack(stack), 0)
    lines = _run_lines(np.nan_to_num(lines).astype(np.float32), np.isnan(lines))
    cv2.imwrite(str(out), np.clip(lines, 0, 255).astype(np.uint8))
    return lines


def _run_lines(img, hole):
    """An etch with the hole's pixels taken from the same row whole line periods away, the nearest first: the
    etch's lines run on through it (a blur would leave a grey patch). The period is read off the rows round the hole
    (autocorrelation along x, LINES_PERIOD px)."""
    ys, xs = np.nonzero(hole)
    if not len(ys):
        return img
    reach = LINES_AROUND
    y0, y1 = max(ys.min() - reach, 0), min(ys.max() + reach + 1, H)
    x0, x1 = max(xs.min() - reach, 0), min(xs.max() + reach + 1, W)
    patch, free = img[y0:y1, x0:x1], ~hole[y0:y1, x0:x1]
    best, period = -2.0, LINES_PERIOD[0]
    for p in range(LINES_PERIOD[0], LINES_PERIOD[1] + 1):
        ok = free[:, :-p] & free[:, p:]
        if ok.sum() < 50:
            continue
        a, b = patch[:, :-p][ok], patch[:, p:][ok]
        a, b = a - a.mean(), b - b.mean()
        r = float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean() + 1e-6))
        if r > best:
            best, period = r, p
    out, left = img.copy(), hole.copy()
    for n in sorted((n for n in range(-LINES_SHIFTS, LINES_SHIFTS + 1) if n), key=lambda n: (abs(n), n)):
        sy, sx = np.nonzero(left)
        src = sx + n * period
        ok = (src >= 0) & (src < W)
        ok[ok] = ~hole[sy[ok], src[ok]]
        out[sy[ok], sx[ok]] = img[sy[ok], src[ok]]
        left[sy[ok], sx[ok]] = False
        if not left.any():
            break
    return out


def _title_symbol(rgb):
    """The title line's symbol as printed (dark on the gold), round ENERGY_SYMBOLS' first. bool HxW"""
    from . import fairy
    cx, cy, r, _ = fairy.ENERGY_SYMBOLS[0]
    yy, xx = np.mgrid[:H, :W]
    near = np.hypot(xx - cx, yy - cy) < r + TITLE_REACH
    return near & (rgb.mean(-1) < TITLE_INK)


def _gold_foil(card, canvas, drawn):
    """The gold print's own foil (shader, plate, etch) for this card. Its footer's print is moved with the card's:
    on the plate the old box and stars give way to the footer's own value, the new box is cut out and the stars' foil
    set where they now stand; on the etch the lines run on under the old print (_gold_lines) and the new box, number
    and stars are etched as the print etches its own. Fairy's title symbol is etched where Psychic's was."""
    template = template_id(card)
    record = foil.printings(template)["base"]
    out = {"front": foil.modern_front(template, canvas), "base": None, "reverse": None}
    alpha = canvas[..., 3] / 255
    x0, y0, x1, y1 = CLEAR["gold"][0]
    zero = np.zeros((H, W), np.float32)
    boxes, number = drawn.get("footer_box", zero), drawn.get("footer", zero)
    stars_now = drawn.get("footer_stars", zero)
    _, cover, sx, sy = gold_stars(card)
    h, w = cover.shape
    dx = int(np.flatnonzero(stars_now.max(axis=0) > 0.5)[0] - (sx + np.flatnonzero(cover.max(axis=0) > 0.5)[0]))

    def moved(img):
        """img with its footer's print off (the footer's own value, row by row) and its stars moved by dx."""
        out = img.copy()
        out[y0:y1, x0:x1] = np.median(img[y0:y1, slice(*FOOT_PLAIN)], axis=1)[:, None]
        region = out[sy:sy + h, sx + dx:sx + dx + w]
        region[:] = region * (1 - cover) + img[sy:sy + h, sx:sx + w] * cover
        return out

    plate = moved(foil.plate(record)) * (1 - boxes)
    tex = foil.texture(record, record.get("_e"))
    if tex is not None and tex.shape[0] == foil.TEX:
        etch = _etch(record)
        lines = _gold_lines()
        new = etch.copy()
        new[y0:y1, x0:x1] = lines[y0:y1, x0:x1]
        region = new[sy:sy + h, sx + dx:sx + dx + w]
        region[:] = region * (1 - cover) + etch[sy:sy + h, sx:sx + w] * cover
        new = np.maximum(new, 255 * np.maximum(boxes, number))
        if card["type"] == "Fairy":                 # Psychic's eye off the title line, Fairy's symbol etched
            new = _run_lines(new, dilate(_title_symbol(tcg.scan(template)[..., :3]), 1))
            new = np.where(_title_symbol(canvas[..., :3]), 0, new)
        tex = tex.copy()
        view = tex[:, tcg.BLEED_X:tcg.BLEED_X + W]
        for ch in range(min(3, view.shape[2])) if view.ndim == 3 else [None]:
            plane = view if ch is None else view[..., ch]
            plane[:] = np.clip(new, 0, 255).astype(plane.dtype)
    spec = (foil.legacy_shader(record.get("_s") or ""), foil.material_name(record.get("_f") or ""),
            record.get("_cc") or None)
    out["base"] = (spec, foil.plate_texture(plate * alpha), {"etch": tex, "cf": foil.texture(record, record.get("_cf"))})
    return out


def _print_foil(card, canvas, print_id=None):
    """The foil of an Energy card that is in the look already: its own TCG Live printing, as it is (shader, plate,
    etch and pattern textures), on the card's own front. print_id: the printing to take instead (a typed Special
    Energy's typed print, base printing only: the holo the Rare prints; its plate foils the swirl round the sphere and
    the header, never what the card sets again)."""
    source = print_id or card["id"]
    out = {"front": foil.modern_front(source, canvas), "base": None, "reverse": None}
    try:
        old = foil.printings(source)
    except ValueError:
        warnings.warn(f"{source}: TCG Live has no foil for this print: the card is set without")
        return out
    alpha = canvas[..., 3] / 255
    for key in ("base",) if print_id else ("base", "reverse"):
        record = old[key]
        if not foil.is_foil(record):
            continue
        spec = (foil.legacy_shader(record.get("_s") or ""), foil.material_name(record.get("_f") or ""),
                record.get("_cc") or None)
        extra = {"etch": foil.texture(record, record.get("_e")), "cf": foil.texture(record, record.get("_cf"))}
        out[key] = (spec, foil.plate_texture(foil.plate(record) * alpha), extra)
    return out
