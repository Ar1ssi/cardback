"""Compose an older regular (non-ex) Pokemon card - Sun & Moon or Sword & Shield - as a Mega Evolution regular card.

The art window of every one of those eras sits where Mega Evolution's does (x 59-674, y 100-484 at 733x1024), so
the old art moves over unscaled; only a Stage card's old evolution portrait and "Evolves from" strip, which
overlap the art, are inpainted first. Card text comes from TCGdex, the Pokedex line from PokeAPI. Layout constants
were measured on me1 scans: name bar, weakness row and footer match the SV ex ones in compose.py; the vertical
spacing of abilities and attacks was fitted on 45 me1 cards.
"""
import re
import warnings

import cv2
import numpy as np

from . import cache, compose, frame, inpaint, meframe, tcg
from . import typeset as ts
from .imaging import dilate, fill_holes, hsv, load_rgba, luminance, save_rgba

W, H = 733, 1024
POKEAPI = "https://pokeapi.co/api/v2"
REGIONAL = {"Galarian": "galar", "Alolan": "alola", "Hisuian": "hisui", "Paldean": "paldea"}
NOT_REGULAR = (" V", " VMAX", " VSTAR", " ex", " EX", " GX", "-GX", " BREAK", " LV.X", " Prism Star")

# Old Sword & Shield Stage cards: evolution portrait box and "Evolves from" strip, where they cover the art.
OLD_STAGE_OVERLAY = [(59, 100, 121, 167), (114, 100, 433, 127)]
OLD_PORTRAIT = (30, 78, 114, 161)                 # inside the silver rim of the old portrait box
PORTRAIT_SIZE = 90

EVOLVES_X, EVOLVES_BASE, EVOLVES_SIZE = 143, 111, 17.5
DEX_X, DEX_BASE, DEX_SIZE, DEX_TRACK = 366, 503.5, 15.0, 0.35
FLAVOR_RIGHT, FLAVOR_LEFT, FLAVOR_SIZE, FLAVOR_PITCH, FLAVOR_CENTRE = 686, 276, 17.6, 22.5, 963.5

# Abilities and attacks (icon-row tops). A lone attack sits at a fixed height; several are stacked with a loose
# gap when they end high enough, otherwise a tight one, and the stack is centred on a 20.5 px grid.
ABILITY_TOP, LONE_ATTACK_TOP, LONE_NO_TEXT_TOP = 555, 638, 659
GRID = (577, 597, 618, 638, 659)
LOOSE_GAP, TIGHT_GAP, LOOSE_MAX_BOTTOM, TIGHT_PREFER_BOTTOM, PANEL_BOTTOM = 61, 33, 806, 825, 865
STACK_CENTRE, PREFERRED_TOP = 690, 597
ROW_H, ATTACK_TO_BODY, ABILITY_TO_BODY, LINE_CAP = 36, 44, 47, 18
NAME_TO_BODY = 41                          # from the name cap top (icon top + 4); 40 on attacks, 42 on abilities
RED_ON_DARK = (compose.WHITE, 1.0, 1.0)     # Spiritomb: red Ability name, thin white keyline
BODY_FIT_TRACK = 0.0                       # a word stays on the line if it fits untracked (Kadabra line 1)


# ---------------------------------------------------------------- card data
def species_line(card):
    """'NO. 0248   Armor Pokemon   HT: 6\'7"   WT: 445.3 lbs.' from PokeAPI (the regional form's height/weight)."""
    dex = (card.get("dexId") or [None])[0]
    if not dex:
        return ""
    sp = cache.fetch_json(f"{POKEAPI}/pokemon-species/{dex}", cache.path("pokeapi", f"species-{dex}.json"))
    genus = next((g["genus"] for g in sp.get("genera", []) if g["language"]["name"] == "en"), "")
    region = next((REGIONAL[p] for p in REGIONAL if card["name"].startswith(p + " ")), None)
    variety = sp["varieties"][0]["pokemon"]["name"]
    if region:
        variety = next((v["pokemon"]["name"] for v in sp["varieties"] if v["pokemon"]["name"].endswith(region)),
                       variety)
    mon = cache.fetch_json(f"{POKEAPI}/pokemon/{variety}", cache.path("pokeapi", f"pokemon-{variety}.json"))
    return dex_text(dex, genus, mon["height"], mon["weight"])


def dex_text(dex, genus, height_dm, weight_hg):
    """The printed Pokedex line; whole feet print without inches (Vulpix: HT: 2')."""
    feet, inch = divmod(round(height_dm * 3.937008), 12)
    lbs = f"{weight_hg * 0.2204623:.1f}".removesuffix(".0")
    height = f"{feet}'" + (f'{inch}"' if inch else "")
    return f"NO. {dex:04d}   {genus}   HT: {height}   WT: {lbs} lbs."


def rarity_symbol(rarity):
    r = (rarity or "").lower()
    if r == "common":
        return "common"
    if r == "uncommon":
        return "uncommon"
    return "rare"


def card_number(local, official):
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", local)
    if m and official:                                   # Shiny Vault and similar: SV33/SV94
        return f"{local}/{m.group(1)}{official}"
    if local.isdigit() and official:
        return f"{int(local):0{len(str(official))}d}/{official}"
    return local


# Sub-sets TCGdex gives no abbreviation of their own; they print their main set's code.
SUBSET_CODES = {"sma": "HIF"}       # Hidden Fates Shiny Vault -> Hidden Fates


def set_code(set_id, s):
    code = ((s.get("abbreviation") or {}).get("official") or s.get("tcgOnline") or SUBSET_CODES.get(set_id, ""))
    if not code:
        warnings.warn(f"set {set_id}: no set code known; the footer's set box is left empty")
    return code.upper()


def read_card(card_id):
    c = tcg.card(card_id)
    name = c.get("name", "")
    if c.get("category") != "Pokemon" or c.get("stage") not in ("Basic", "Stage1", "Stage2") \
            or name.endswith(NOT_REGULAR):
        raise SystemExit(f"{card_id} is {name!r} ({c.get('stage')}): only regular Basic/Stage 1/Stage 2 Pokemon")
    types = c.get("types") or []
    if not types or types[0] not in meframe.POOL:
        raise SystemExit(f"{card_id}: no Mega Evolution frame for type {types[:1]}")
    s = tcg.set_info(c["set"]["id"])
    official = (s.get("cardCount") or {}).get("official")
    prefix = next((p for p in compose.PREFIXES if name.startswith(p + " ")), "")
    printed = lambda t: (t or "").replace("'", "’")
    for b in (c.get("abilities") or []) + (c.get("attacks") or []):
        b["effect"] = printed(b.get("effect"))
        b["name"] = printed(b.get("name"))
    return {
        "id": card_id, "type": types[0], "prefix": prefix, "name": name[len(prefix):].strip(),
        "hp": str(c.get("hp", "")), "stage": {"Basic": 0, "Stage1": 1, "Stage2": 2}[c["stage"]],
        "evolves_from": printed(c.get("evolveFrom")), "abilities": c.get("abilities") or [],
        "attacks": c.get("attacks") or [], "weakness": (c.get("weaknesses") or [None])[0],
        "resistance": (c.get("resistances") or [None])[0], "retreat": int(c.get("retreat") or 0),
        "illustrator": c.get("illustrator") or "", "reg_mark": c.get("regulationMark") or "",
        "set_code": set_code(c["set"]["id"], s),
        "number": card_number(c.get("localId", ""), official), "year": (s.get("releaseDate") or "")[:4],
        "rarity": rarity_symbol(c.get("rarity")), "flavor": printed(c.get("description")),
        "dex_line": species_line(c),
        # Mega Evolution cards (used to check the layout against the real print) keep their own portrait.
        "old_stage": c["stage"] != "Basic" and not card_id.startswith("me"),
    }


# ---------------------------------------------------------------- art
def old_art(card):
    """The old card's art window (733x1024 canvas, only the window filled), with a Stage card's portrait and
    strip inpainted; cached."""
    out = cache.path("art", f"me-{card['id']}.png")
    if out.exists():
        return load_rgba(out)
    scan = tcg.scan(card["id"])
    x0, y0, x1, y1 = meframe.ART
    if card["old_stage"]:
        mask = np.zeros((H, W), bool)
        for bx0, by0, bx1, by1 in OLD_STAGE_OVERLAY:
            mask[by0:by1, bx0:bx1] = True
        window = np.zeros((H, W), bool)
        window[y0:y1, x0:x1] = True
        # LaMa sees only the art window: the old frame around it must not leak into the fill.
        work = scan.copy()
        work[~window, :3] = 0
        work = inpaint.inpaint(work[y0 - 8:y1 + 8, x0 - 8:x1 + 8], (mask & window)[y0 - 8:y1 + 8, x0 - 8:x1 + 8])
        scan[y0:y1, x0:x1] = work[8:-8, 8:-8]
    art = np.zeros_like(scan)
    art[y0:y1, x0:x1] = scan[y0:y1, x0:x1]
    save_rgba(art, out)
    return art


def portrait_sprite(card):
    """The pre-evolution picture from the old portrait box, its silver background keyed out. -> (rgb, alpha)"""
    x0, y0, x1, y1 = OLD_PORTRAIT
    crop = tcg.scan(card["id"])[y0:y1, x0:x1, :3]
    _, sat, val = hsv(crop)
    plain = (sat < 0.14) & (val > 0.6)
    n, labels = cv2.connectedComponents(plain.astype(np.uint8), connectivity=4)
    edge = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    background = np.isin(labels, [i for i in edge if i != 0]) & plain
    alpha = cv2.GaussianBlur((~background).astype(np.float32), (3, 3), 0.7)
    size = (PORTRAIT_SIZE, round(PORTRAIT_SIZE * crop.shape[0] / crop.shape[1]))
    return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC), cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)


def put_portrait(canvas, card):
    rgb, alpha = portrait_sprite(card)
    h, w = alpha.shape
    cx, cy, r = meframe.PORTRAIT_DISC
    x0, y0 = round(cx - w / 2), round(cy - h / 2)
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    disc = np.clip(r - 0.5 - np.hypot(xx - cx, yy - cy), 0, 1)
    a = (alpha * disc)[..., None]
    region = canvas[y0:y0 + h, x0:x0 + w, :3]
    region[:] = region * (1 - a) + rgb * a


# ---------------------------------------------------------------- abilities and attacks
def _extent(kind, lines, pitch):
    """Height of a block from its icon-row top to its last baseline (or the icon row's bottom)."""
    if kind == "ability":
        return ABILITY_TO_BODY + pitch * (max(1, len(lines)) - 1) + LINE_CAP
    if lines:
        return ATTACK_TO_BODY + pitch * (len(lines) - 1) + LINE_CAP
    return ROW_H


def _snap(top):
    return min(GRID, key=lambda g: abs(g - top))


def me_layout(laid, pitch):
    """Vertical placement of abilities then attacks, fitted on me1. laid: [(kind, block, lines)].
    -> (top of each block's name cap, as compose.text_block expects; whether it fits)"""
    extents = [_extent(kind, lines, pitch) for kind, _, lines in laid]

    def stack(first, gap):
        tops, t = [], first
        for e in extents:
            tops.append(t)
            t += e + gap
        return tops, tops[-1] + extents[-1]

    if laid[0][0] == "ability":
        tops, bottom = stack(ABILITY_TOP, LOOSE_GAP)
        if bottom > LOOSE_MAX_BOTTOM:
            tops, bottom = stack(ABILITY_TOP, TIGHT_GAP)
    elif len(laid) == 1:
        tops, bottom = stack(LONE_ATTACK_TOP if laid[0][2] else LONE_NO_TEXT_TOP, 0)
    else:
        span = sum(extents) + LOOSE_GAP * (len(laid) - 1)
        tops, bottom = stack(max(PREFERRED_TOP, _snap(STACK_CENTRE - span / 2)), LOOSE_GAP)
        if bottom > LOOSE_MAX_BOTTOM:
            span = sum(extents) + TIGHT_GAP * (len(laid) - 1)
            first = _snap(STACK_CENTRE - span / 2)
            if first < PREFERRED_TOP and PREFERRED_TOP + span <= TIGHT_PREFER_BOTTOM:
                first = PREFERRED_TOP
            tops, bottom = stack(first, TIGHT_GAP)
    # compose.text_block works from the name's cap top: icon-row top + 4 (attacks), badge top + 5 (abilities).
    caps = [t + (5 if kind == "ability" else 4) for t, (kind, _, _) in zip(tops, laid)]
    return caps, bottom <= PANEL_BOTTOM


# ---------------------------------------------------------------- text
def name_and_hp(canvas, card, prof):
    compose.name_bar_text(canvas, card, prof, ex=False, name_right_limit=meframe.NAME_RIGHT_LIMIT)


def evolves_from(canvas, card, ink):
    ts.stamp(canvas, ts.coverage([dict(text=f"Evolves from {card['evolves_from']}", face=ts.BOLD_ITALIC,
                                       size=EVOLVES_SIZE, x=EVOLVES_X, base=EVOLVES_BASE)]), ink)


def dex_line(canvas, card):
    if card["dex_line"]:
        ts.stamp(canvas, ts.coverage([dict(text=card["dex_line"], face=ts.REGULAR, size=DEX_SIZE, x=DEX_X,
                                           base=DEX_BASE, tracking=DEX_TRACK, anchor="c")]),
                 np.array(frame.DARK_INK, np.float32))


def wrap(text, face, size, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and ts.text_width(trial, face, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def flavor(canvas, card, ink):
    if not card["flavor"]:
        return
    size = FLAVOR_SIZE
    lines = wrap(card["flavor"], ts.REGULAR, size, FLAVOR_RIGHT - FLAVOR_LEFT)
    while len(lines) > 3 and size > 12:
        size -= 0.5
        lines = wrap(card["flavor"], ts.REGULAR, size, FLAVOR_RIGHT - FLAVOR_LEFT)
    if len(lines) > 3:
        warnings.warn(f"{card['id']}: flavor text needs {len(lines)} lines")
    pitch = FLAVOR_PITCH * size / FLAVOR_SIZE
    first = FLAVOR_CENTRE - pitch * (len(lines) - 1) / 2
    ts.stamp(canvas, ts.coverage([dict(text=t, face=ts.REGULAR, size=size, x=FLAVOR_RIGHT, base=first + i * pitch,
                                       anchor="r") for i, t in enumerate(lines)]), ink)


# ---------------------------------------------------------------- card
def compose_card(card_id, attacks=True, card=None):
    """-> (RGBA float32 733x1024, report dict). attacks=False leaves out abilities and attacks; card: read_card's
    dict, possibly edited (default: read it)."""
    card = card or read_card(card_id)
    prof = frame.profile(card["type"])
    canvas = meframe.template(card["type"]).copy()
    art = old_art(card)
    x0, y0, x1, y1 = meframe.ART
    canvas[y0:y1, x0:x1, :3] = art[y0:y1, x0:x1, :3]
    if card["stage"]:
        sprite = meframe.stage_sprite(card["stage"])
        a = sprite[..., 3:4] / 255
        canvas[..., :3] = canvas[..., :3] * (1 - a) + sprite[..., :3] * a
        if card["old_stage"]:
            put_portrait(canvas, card)

    panel_lum = float(np.median(luminance(canvas[meframe.PANEL[1]:meframe.PANEL[3], meframe.PANEL[0]:meframe.PANEL[2]])))
    white_text = panel_lum < 110
    fill = compose.WHITE if white_text else np.array(frame.DARK_INK, np.float32)
    name_and_hp(canvas, card, prof)
    if card["stage"]:
        evolves_from(canvas, card, np.array(frame.DARK_INK, np.float32))
    dex_line(canvas, card)
    shrink = None if not attacks else compose.text_block(canvas, card, fill, None, layout=me_layout, name_to_body=NAME_TO_BODY,
                                fit_tracking=BODY_FIT_TRACK, red_edge=RED_ON_DARK if white_text else None)
    compose.weakness_row(canvas, card, fill, None, labels=False)
    compose.footer(canvas, card, white_text, rarity=card["rarity"])
    flavor(canvas, card, fill)
    return canvas, {"card": card, "body_scale": shrink}
