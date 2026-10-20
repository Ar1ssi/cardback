"""Compose a Sword & Shield Basic V as a Scarlet & Violet ex card.

Card text comes from TCGdex. Layout constants were measured at 733x1024 on real SV scans: Absol ex (name bar,
attacks, weakness row), Paldean Clodsire ex (regional-form name), Mew ex and Chien-Pao ex (ability rows),
Wo-Chien ex (four-cost attacks, damage), Alakazam ex (footer, resistance).
"""
import re
import warnings
from dataclasses import dataclass

import numpy as np

from . import cache, frame, refs, swsh, tcg
from . import typeset as ts
from .imaging import load_rgba, luminance, put_sprite, save_rgba

W, H = 733, 1024
WHITE = np.array([255, 255, 255], np.float32)
NAME_EDGE = ([8, 32, 38], 1.0, 0.55)          # soft dark edge under white name-bar text (Absol)
ART_EDGE = ([14, 14, 18], 1.7, 0.95)          # keyline for white text set directly on the art (hybrid)
LIGHT_KEYLINE = (WHITE, 1.7, 1.0)              # white keyline under dark footer text (Alakazam)
DARK_KEYLINE = ([20, 20, 24], 1.7, 1.0)        # dark keyline under white footer text (Absol)
ABILITY_RED = np.array([192, 26, 32], np.float32)
COPYRIGHT_GREY = np.array([52, 52, 56], np.float32)

PREFIXES = ("Galarian", "Alolan", "Hisuian", "Paldean", "Origin Forme", "Rapid Strike", "Single Strike",
            "Fusion Strike", "Ice Rider", "Shadow Rider", "Dusk Mane", "Dawn Wings")

NAME_BASE, NAME_X, PREFIX_RATIO, NAME_SIZE, NAME_MAX_XSCALE, NAME_MIN_XSCALE = 78, 141, 0.62, 49.5, 0.89, 0.6
EX_LOGO_GAP, EX_LOGO_RIGHT_LIMIT = 5, 462
HP_SIZE, HP_XSCALE, HP_RIGHT, HP_BASE = 64.0, 0.86, 635, 85

ATK_FACE, ATK_SIZE, ATK_XSCALE, ATK_X = ts.COND_BOLD, 41.0, 0.885, 237
ABILITY_SIZE, ABILITY_XSCALE, ABILITY_X, BADGE_X = 39.5, 0.90, 238, 55
DMG_SIZE, DMG_XSCALE, DMG_RIGHT = 41.0, 0.80, 672
COST_X0, COST_PITCH, COST_R = 71, 44, 18.8
BODY_FACE, BODY_SIZE, BODY_X, BODY_FULL_W, BODY_TRACK = ts.MEDIUM, 24.9, 60, 615, 0.6
BODY_PITCH, NAME_TO_BODY, BLOCK_GAP, NAME_ROW_H = 29, 45, 51, 30
BLOCK_CENTRE, BLOCK_TOP_MIN, BLOCK_BOTTOM_MAX = (597 + 794) / 2, 540, 862
SHRINK_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8)

ROW_ICON_R, ROW_Y = 12.8, 895.6
WEAK_ICON_X, RES_ICON_X, RETREAT_X0, RETREAT_PITCH = 126.7, 320.5, 540.2, 30
BAR_ROWS = slice(487, 522)


@dataclass
class Options:
    style: str = "hybrid"        # "hybrid": no panel over the art; "panel": the SV translucent type panel
    attacks: bool = True         # False: no ability/attack block (names, text, costs, damage)


# ---------------------------------------------------------------- card data
def read_card(card_id):
    c = tcg.card(card_id)
    name = c.get("name", "")
    if c.get("category") != "Pokemon" or c.get("stage") != "Basic" or not name.endswith(" V"):
        raise SystemExit(f"{card_id} is {name!r} ({c.get('stage')}): only Sword & Shield Basic Pokemon V are supported")
    types = c.get("types") or []
    if not types or types[0] not in refs.TYPES:
        raise SystemExit(f"{card_id}: no Scarlet & Violet frame for type {types[:1]}")
    s = tcg.set_info(c["set"]["id"])
    official = (s.get("cardCount") or {}).get("official") or (c["set"].get("cardCount") or {}).get("official")
    local = c.get("localId", "")
    number = f"{int(local):0{len(str(official))}d}/{official}" if local.isdigit() and official else local
    base = name[:-2]
    prefix = next((p for p in PREFIXES if base.startswith(p + " ")), "")
    printed = lambda s: (s or "").replace("'", "’")      # TCGdex uses straight apostrophes; cards print ’
    for b in (c.get("abilities") or []) + (c.get("attacks") or []):
        b["effect"] = printed(b.get("effect"))
        b["name"] = printed(b.get("name"))
    return {
        "id": card_id, "type": types[0], "prefix": prefix, "name": base[len(prefix):].strip(), "hp": str(c.get("hp", "")),
        "abilities": c.get("abilities") or [], "attacks": c.get("attacks") or [],
        "weakness": (c.get("weaknesses") or [None])[0], "resistance": (c.get("resistances") or [None])[0],
        "retreat": int(c.get("retreat") or 0), "illustrator": c.get("illustrator") or "",
        "reg_mark": c.get("regulationMark") or "", "set_code": ((s.get("abbreviation") or {}).get("official")
                                                              or s.get("tcgOnline") or "").upper(),
        "number": number, "year": (s.get("releaseDate") or "")[:4],
    }


def clean_art(card_id):
    """The card's art with the SWSH frame and text rebuilt; cached (LaMa takes ~15 s)."""
    out = cache.path("art", f"{card_id}.png")
    if out.exists():
        return load_rgba(out), None
    art, mask, report = swsh.clean(tcg.scan(card_id))
    save_rgba(art, out)
    return art, report


# ---------------------------------------------------------------- layers
def panel(art, gradient):
    """SV type panel over the lower art: the art's brightness, rank for rank, mapped onto the reference panel's
    colours (darkest to brightest), so the art shows through in the type's own palette."""
    canvas = art.copy()
    rows, cols = slice(487, 996), slice(28, 704)
    lum = luminance(art[rows, cols])
    ranks = lum.ravel().argsort().argsort().reshape(lum.shape) / max(1, lum.size - 1)
    pos = ranks * (len(gradient) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(gradient) - 1)
    t = (pos - lo)[..., None]
    canvas[rows, cols, :3] = gradient[lo] * (1 - t) + gradient[hi] * t
    return canvas


def name_bar_text(canvas, card, prof, ex=True, name_right_limit=EX_LOGO_RIGHT_LIMIT):
    """Name (regional prefix smaller), ex logo when ex, and HP. -> horizontal squeeze applied to the name."""
    ink = prof["ink"]
    edge = NAME_EDGE if prof["light_text"] else None
    logo, logo_left, logo_right = frame.ex_logo()
    logo_room = (logo_right - logo_left) + EX_LOGO_GAP if ex else 0
    small = NAME_SIZE * PREFIX_RATIO
    gap = 8 if card["prefix"] else 0
    words = ts.text_width(card["prefix"], ts.BOLD, small) + ts.text_width(card["name"], ts.BOLD, NAME_SIZE)
    room = name_right_limit - logo_room - NAME_X - gap
    xscale = min(NAME_MAX_XSCALE, room / words)
    if xscale < NAME_MIN_XSCALE:
        warnings.warn(f"name {card['prefix']} {card['name']!r} squeezed to {xscale:.2f} width to fit the bar")
    x_big = NAME_X + ts.text_width(card["prefix"], ts.BOLD, small, xscale) + gap
    name_right = x_big + ts.text_width(card["name"], ts.BOLD, NAME_SIZE, xscale)
    ts.stamp(canvas, ts.coverage([
        dict(text=card["prefix"], face=ts.BOLD, size=small, x=NAME_X, base=NAME_BASE, xscale=xscale),
        dict(text=card["name"], face=ts.BOLD, size=NAME_SIZE, x=x_big, base=NAME_BASE, xscale=xscale),
    ]), ink, edge)
    if ex:
        ox, oy = logo[4]
        put_sprite(canvas, logo, ox + name_right + EX_LOGO_GAP - logo_left, oy)   # same height as on Absol
    hp_w = ts.text_width(card["hp"], ts.COND_BOLD, HP_SIZE, HP_XSCALE)
    ts.stamp(canvas, ts.coverage([
        dict(text=card["hp"], face=ts.COND_BOLD, size=HP_SIZE, x=HP_RIGHT, base=HP_BASE, xscale=HP_XSCALE, anchor="r"),
        dict(text="HP", face=ts.COND_BOLD, size=19.4, x=HP_RIGHT - hp_w - 4, base=84, anchor="r"),
    ]), ink, edge)
    return xscale


def energy(type_name, cx, cy, r, canvas):
    prof = frame.profile(type_name)
    put_sprite(canvas, prof["icon"], cx, cy, scale=r / prof["icon_r"])


def sv_ex_layout(laid, pitch):
    """Absol ex: blocks stacked with a fixed gap and centred on the panel. laid: [(kind, block, lines)].
    -> (top of each block's name cap, whether it fits)"""
    heights = [(NAME_TO_BODY + pitch * (len(lines) - 1)) if lines else NAME_ROW_H for _, _, lines in laid]
    span = sum(heights) + BLOCK_GAP * (len(laid) - 1)
    top = round(BLOCK_CENTRE - span / 2)
    tops = []
    for h in heights:
        tops.append(top)
        top += h + BLOCK_GAP
    return tops, BLOCK_TOP_MIN <= tops[0] and tops[0] + span <= BLOCK_BOTTOM_MAX - pitch


def text_block(canvas, card, fill, edge, layout=sv_ex_layout, name_to_body=NAME_TO_BODY, fit_tracking=None,
               red_edge=None):
    """Abilities then attacks, placed by layout(laid, pitch); body text shrinks until the layout fits.
    red_edge: keyline under the red Ability names (default: a white one when the text has an edge)."""
    blocks = [("ability", a) for a in card["abilities"]] + [("attack", a) for a in card["attacks"]]
    if not blocks:
        return None
    for k in SHRINK_STEPS:
        size, pitch = BODY_SIZE * k, round(BODY_PITCH * k)
        laid = [(kind, b, ts.justify(b.get("effect") or "", size, BODY_FULL_W, BODY_TRACK, fit_tracking)
                 if b.get("effect") else []) for kind, b in blocks]
        tops, fits = layout(laid, pitch)
        if fits:
            break
    else:
        warnings.warn(f"{card['id']}: abilities and attacks run taller than the panel holds")
    atk_cap = ts.cap_height(ATK_FACE, ATK_SIZE)
    body_cap = ts.cap_height(BODY_FACE, size)
    items, red_items, icons = [], [], []
    for (kind, b, lines), top in zip(laid, tops):
        if kind == "ability":
            badge, (bw, bh) = frame.ability_badge()
            put_sprite(canvas, badge, BADGE_X, top - 5)     # Chien-Pao: badge rim top 551, name cap top 556
            red_items.append(dict(text=b["name"], face=ATK_FACE, size=ABILITY_SIZE, x=ABILITY_X,
                                  base=top + ts.cap_height(ATK_FACE, ABILITY_SIZE), xscale=ABILITY_XSCALE))
        else:
            cost = b.get("cost") or []
            for i, t in enumerate(cost):
                icons.append((t, COST_X0 + COST_PITCH * i, top + 14))
            name_x = max(ATK_X, COST_X0 + COST_PITCH * (len(cost) - 1) + COST_R + 14) if cost else ATK_X
            items.append(dict(text=b["name"], face=ATK_FACE, size=ATK_SIZE, x=name_x, base=top + atk_cap,
                              xscale=ATK_XSCALE))
            dmg = str(b.get("damage") or "")
            m = re.fullmatch(r"(\d+)([+×x-]?)", dmg)
            if m:
                digits, suffix = m.groups()
                items.append(dict(text=digits, face=ATK_FACE, size=DMG_SIZE, x=DMG_RIGHT, base=top + atk_cap,
                                  xscale=DMG_XSCALE, anchor="r"))
                if suffix:
                    items.append(dict(text="×" if suffix in "×x" else suffix, face=ATK_FACE, size=DMG_SIZE,
                                      x=DMG_RIGHT + 1, base=top + atk_cap, xscale=DMG_XSCALE))
        body_top = top + name_to_body
        for i, (runs, tracking) in enumerate(lines):
            items += ts.line_items(runs, tracking, size, BODY_X, body_top + i * pitch + body_cap)
    ts.stamp(canvas, ts.coverage(items), fill, edge)
    if red_items:
        ts.stamp(canvas, ts.coverage(red_items), ABILITY_RED, red_edge or (LIGHT_KEYLINE if edge else None))
    for t, cx, cy in icons:
        energy(t, cx, cy, COST_R, canvas)
    return size / BODY_SIZE


def weakness_row(canvas, card, fill, edge, labels=True):
    label = 22.0 * 69 / ts.text_width("weakness", ts.MEDIUM, 22.0)      # Absol's "weakness" ink is 69 px wide
    items = [dict(text="weakness", face=ts.MEDIUM, size=label, x=41, base=901),
             dict(text="resistance", face=ts.MEDIUM, size=label, x=229, base=901),
             dict(text="retreat", face=ts.MEDIUM, size=label, x=457, base=901)] if labels else []
    if card["weakness"]:
        value = card["weakness"].get("value", "×2").replace("x", "×")
        if value.startswith("+"):                    # older cards: +20
            items.append(dict(text=value, face=ts.COND_BOLD, size=31.5, x=146, base=907))
        else:
            items += [dict(text="×", face=ts.MEDIUM, size=27.0, x=146, base=905),
                      dict(text=value.lstrip("×"), face=ts.COND_BOLD, size=31.5, x=161, base=907)]
    if card["resistance"]:
        value = card["resistance"].get("value", "-30").replace("-", "−")
        items.append(dict(text=value, face=ts.COND_BOLD, size=31.5, x=339, base=908))
    ts.stamp(canvas, ts.coverage(items), fill, edge)
    if card["weakness"]:
        energy(card["weakness"]["type"], WEAK_ICON_X, ROW_Y, ROW_ICON_R, canvas)
    if card["resistance"]:
        energy(card["resistance"]["type"], RES_ICON_X, ROW_Y, ROW_ICON_R, canvas)
    for i in range(card["retreat"]):
        energy("Colorless", RETREAT_X0 + RETREAT_PITCH * i, ROW_Y, ROW_ICON_R, canvas)


def rarity_mark(symbol, x, y=980.5):
    """Coverage of the rarity symbol(s) starting at x: 'common' circle, 'uncommon' diamond, 'rare' star,
    'double' two stars (Double Rare, as on ex)."""
    if symbol == "common":
        yy, xx = np.mgrid[:H, :W]
        return np.clip(5.6 - np.hypot(xx - (x + 6), yy - y), 0, 1).astype(np.float32)
    if symbol == "uncommon":
        yy, xx = np.mgrid[:H, :W]
        return np.clip(6.8 - (np.abs(xx - (x + 6.5)) + np.abs(yy - y)), 0, 1).astype(np.float32)
    stars = ts.star(x + 7, y, 7.6, 3.1)
    if symbol == "double":
        stars = np.maximum(stars, ts.star(x + 24, y, 7.6, 3.1))
    return stars


def footer(canvas, card, white_text, rarity="double"):
    ink, keyline = (WHITE, DARK_KEYLINE) if white_text else (np.array(frame.DARK_INK, np.float32), LIGHT_KEYLINE)
    dark = np.array([38, 33, 38], np.float32)
    ts.stamp(canvas, ts.coverage([dict(text=f"Illus. {card['illustrator']}", face=ts.BOLD_ITALIC, size=15.5, x=46,
                                       base=953, xscale=0.97)]), ink, keyline)
    if card["reg_mark"]:
        ts.stamp(canvas, ts.rounded_box(44.5, 960.5, 62.5, 987.5, 3), dark)
        ts.stamp(canvas, ts.rounded_box(45.6, 961.6, 61.4, 986.4, 2.2), WHITE)
        ts.stamp(canvas, ts.coverage([dict(text=card["reg_mark"], face=ts.COND_BOLD, size=19.5, x=53.6, base=981,
                                           anchor="c")]), dark)
    if card["set_code"]:
        code_w = ts.text_width(card["set_code"], ts.COND_BOLD, 19.5, 0.8)
        en_w = ts.text_width("EN", ts.COND_BOLD, 11.5, 0.85)
        box_x1 = max(115.5, 73 + code_w + 1.5 + en_w + 3)
        ts.stamp(canvas, ts.rounded_box(67.5, 960.5, box_x1, 987.5, 3.2), dark)
        ts.stamp(canvas, ts.rounded_box(69.0, 962.0, box_x1 - 1.5, 986.0, 2.4), WHITE)
        ts.stamp(canvas, ts.rounded_box(70.2, 963.2, box_x1 - 2.7, 984.8, 1.8), [28, 24, 28])
        left = (67.5 + box_x1) / 2 - (code_w + 1.5 + en_w) / 2
        ts.stamp(canvas, ts.coverage([dict(text=card["set_code"], face=ts.COND_BOLD, size=19.5, x=left, base=981,
                                           xscale=0.8),
                                      dict(text="EN", face=ts.COND_BOLD, size=11.5, x=left + code_w + 1.5, base=981,
                                           xscale=0.85)]), WHITE)
        num_x = box_x1 + 8.5
    else:                                   # promos print no set code: the number follows the regulation mark
        num_x = 71.0 if card["reg_mark"] else 46.0
    ts.stamp(canvas, ts.coverage([dict(text=card["number"], face=ts.HEAVY_ITALIC, size=16.6, x=num_x, base=988,
                                       xscale=1.11)]), ink, keyline)
    num_right = num_x + ts.text_width(card["number"], ts.HEAVY_ITALIC, 16.6, 1.11)
    ts.stamp(canvas, rarity_mark(rarity, num_right + 6), ink, keyline)
    ts.stamp(canvas, ts.coverage([dict(text=f"©{card['year']} Pokémon / Nintendo / Creatures / GAME FREAK",
                                       face=ts.REGULAR, size=12.6, x=367, base=1011, anchor="c")]), COPYRIGHT_GREY)


def compose(card_id, opts, card=None):
    """-> (RGBA float32 733x1024, report dict). card: read_card's dict, possibly edited (default: read it)."""
    card = card or read_card(card_id)
    art, clean_report = clean_art(card_id)
    prof = frame.profile(card["type"])
    hybrid = opts.style == "hybrid"
    canvas = art.copy() if hybrid else panel(art, prof["gradient"])

    common = frame.common_frame()
    fa = common[..., 3:4] / 255
    if hybrid:
        fa = fa.copy()
        fa[BAR_ROWS, 28:704] = 0                       # no window bar without a panel to divide from
    canvas[..., :3] = canvas[..., :3] * (1 - fa) + common[..., :3] * fa
    bar = frame.name_bar(card["type"])
    ba = bar[..., 3:4] / 255
    canvas[..., :3] = canvas[..., :3] * (1 - ba) + bar[..., :3] * ba
    canvas[..., 3] = tcg.scan(refs.ABSOL.card_id, refs.ABSOL.sha)[..., 3]     # rounded corners

    name_xscale = name_bar_text(canvas, card, prof)
    white_text = hybrid or prof["light_text"]
    fill = WHITE if white_text else prof["ink"]
    edge = ART_EDGE if hybrid else None
    shrink = text_block(canvas, card, fill, edge) if opts.attacks else None
    weakness_row(canvas, card, fill, edge)
    footer(canvas, card, white_text)
    return canvas, {"card": card, "name_xscale": round(name_xscale, 3), "body_scale": shrink,
                    "art_detection": clean_report}
