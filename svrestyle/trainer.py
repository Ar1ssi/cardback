"""Compose an older Trainer card - Item, Supporter, Stadium or Pokemon Tool, from Sword & Shield back to the first
sets - as a Scarlet & Violet / Mega Evolution Trainer.

The Gen 9 frame is trainerframe.template(kind): every SV / ME Trainer of a kind prints the same one. The old art
window moves into the new one: it sits where the new one does on Sword & Shield and Sun & Moon cards, a little
smaller on XY, Black & White and HeartGold SoulSilver cards (each era's window is fixed, ERA_ART), and is found on
older ones by registering the card's frame to its layout's (oldwindow); it is scaled to cover the new window.

Card text comes from TCGdex, with the old eras' rule sentences taken out (the pill prints the rule) and spelled-out
types printed as energy symbols, as the newer cards print them. Layout constants were measured on 151 SV / ME
Trainers: name, Supporter's character, effect text (size, justification, the 20.2 px grid its first line snaps to),
Technical Machine attacks, fossils' HP and Ability, and footer.
"""
import re
import warnings

import cv2
import numpy as np

from . import badges, cache, compose, edit, frame, layers, mecompose, modern, oldwindow, tcg, trainerframe
from . import typeset as ts
from .imaging import load_rgba, save_rgba

W, H = 733, 1024
ART = trainerframe.ART
ART_VERSION = "2"           # of the cached art: bump when how it is cut or cleaned changes

# TCGdex trainer type -> the Gen 9 frame it prints in. Technical Machines are Pokemon Tools since Sword & Shield;
# Rocket's Secret Machines are played like Items. The first sets print no type ("Trainer"): Item, as reprints are.
KINDS = {"Item": "Item", "Supporter": "Supporter", "Stadium": "Stadium", "Tool": "Tool",
         "Technical Machine": "Tool", "Rocket's Secret Machine": "Item", None: "Item"}
# Full-art and gold prints have no art window to move (matched in any case: TCGdex writes "Special illustration
# rare"); ACE SPEC cards print a rule the four Gen 9 frames have no place for.
NOT_REGULAR = ("ultra", "secret", "rainbow", "illustration", "hyper", "shiny", "full art", "gold", "amazing",
               "radiant", "trainer gallery", "double rare", "ace spec")
# Black & White's ACE SPEC cards, which TCGdex rates plain "Rare" (their red ACE SPEC band checked on the scans).
ACE_SPEC = {"bw7-137", "bw7-138", "bw7-139", "bw7-140", "bw8-128", "bw8-129", "bw8-130", "bw9-107", "bw9-108",
            "bw10-92", "bw10-93", "bw10-94", "bw10-95"}
PRISM = "◇"
# A card played either way up (xy10-94 Chaos Tower) prints its other half upside down in the art.
TWO_WAY = "which way this card faces"
# Z-Crystals that give a GX attack (sm11-195, sm12-190): the GX rule has no place on the Gen 9 frames either.
GX_ATTACK = "GX attack on this card"
# Team Flare Hyper Gear (xy4-97 Head Ringer) attaches to the opponent's Pokemon-EX: a rule in its red box that TCGdex
# leaves out and the Gen 9 Tool pill contradicts.
HYPER_GEAR = "Team Flare Hyper Gear"
# Celebrations' Classic Collection reprints each card in its first frame, under ids no scan source maps.
CLASSIC_SET = "cel25cc"
POCKET_SERIES = "tcgp"     # Pokemon TCG Pocket: cards that were never printed

# The old card's art inside its rim (x0, y0, x1, y1), per era, measured on 24 prints each (their art differs, their
# frame does not); the rim's shading left out.
ERA_ART = {
    "swsh": (61, 148, 675, 530),
    "sm": (60, 149, 675, 530),
    "xy": (76, 167, 657, 523),
    "bw": (76, 167, 657, 523),
    "hgss": (71, 161, 662, 543),
}
# Black & White and XY Pokemon Tools print their "Pokemon Tool" band over the art's foot (from y 515, 5 prints).
ERA_TOOL_ART = {"xy": (76, 167, 657, 513), "bw": (76, 167, 657, 513)}
ERAS = {    # TCGdex set id prefixes per era; sv and me are Gen 9 already (their own window is the new one)
    "swsh": ("swsh", "cel25", "pgo", "fut2020"), "sm": ("sm", "det1"), "xy": ("xy", "g1", "dc1"),
    "bw": ("bw", "dv1", "rc"), "hgss": ("hgss", "col1"),
}
# Name: Gill Sans Bold narrowed, as on the Pokemon cards; a long name narrows further to fit the bar, or the room left
# of the Supporter's character.
NAME_SIZE, NAME_XSCALE, NAME_MIN_XSCALE, NAME_X, NAME_BASE = 50.0, 0.89, 0.6, 46, 119
NAME_RIGHT = 690
# Technical Machines print their long names a little larger and lower, narrowed to end by TM_NAME_RIGHT (sv04
# Blindside, Evolution, Devolution, Turbo Energize; sv04.5 Crisis Punch: within 7 px).
TM_PREFIX, TM_NAME_SIZE, TM_NAME_XSCALE, TM_NAME_BASE, TM_NAME_RIGHT = "Technical Machine", 51.5, 0.845, 124.5, 660
CHARACTER_FACE, CHARACTER_SIZE, CHARACTER_RIGHT, CHARACTER_BASE = ts.BOLD_ITALIC, 25.0, 687, 116
CHARACTER_INK = np.array([102, 103, 107], np.float32)
CHARACTER_GAP = 24         # px kept between the name and the character
CHARACTER_MAX_W = 260      # px a character may take (an edited one is narrowed); "Professor Magnolia" takes 200
# Footer: the Pokemon cards' (compose.footer), each part a little off where they print it (fitted on 20 prints).
FOOTER_SHIFT = {"illus": (0, -1), "boxes": (0, -1), "number": (-1, -2), "rarity": (4, -1), "copyright": (-1, 1)}

# Effect text: justified by letter-spacing like the Pokemon cards', but broken for the paragraph as a whole
# (typeset.compose_paragraph, no line squeezed below BODY_MIN_TRACK), a number staying with the word after it;
# paragraphs are a blank line apart. Size, spacing and breaks fitted on the text panels of 85 prints (the breaks of
# 95 of the 115 that a rule decides; greedy breaking gets 81). The first baseline snaps to a 20.2 px grid at the
# line count's centring, fitted on the first lines of 116 prints (1 line: 708, 2-3: 688, 4: 667, 5-6: 647, 8: 607).
BODY_FACE, BODY_SIZE, BODY_X, BODY_FULL_W, BODY_TRACK, BODY_MIN_TRACK = ts.MEDIUM, 26.25, 55, 623, 0.1, -0.6
BODY_PITCH = 35
# A choice's options: "• " at the text's left, the option hanging BULLET_INDENT px in, the options run on without a
# blank line between them (sv06-154 Kieran: bullet ink at 58, the text's at 76; the lines 35 px apart).
BULLET, BULLET_INDENT = "•", 18
GRID_TOP, GRID_STEP = 708, 20.2
BLOCK_CENTRE, ROW_HALF = 711, 14.45
SYMBOL_DROP = 5            # px the block sits lower when its first line holds an energy symbol (taller than a cap;
                           # 15 prints, symbols further down move nothing)
BODY_TOP_MIN, BODY_BOTTOM_MAX = 565, 868          # first and last baseline allowed
SHRINK_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8)
PANEL_BOTTOM = 880
# Technical Machines: the effect from a fixed first baseline, then the attacks (cap top of the name TM_GAP below the
# effect's last baseline), set like the Mega Evolution cards' attacks.
TM_FIRST_BASE, TM_GAP, TM_NAME_TO_BODY, TM_COST_DY = 583, 41, 43, 11.7
BLOCK_ROOM = 70            # px an attack or Ability with a line of text needs
# Fossils and dolls, played as Basic Pokemon: the HP at the bar's right (the Pokemon cards' figures, moved), the
# effect set tighter (a paragraph's last line at -1.04, a one-line paragraph squeezed to -1.34) and run on without a
# blank line between paragraphs, then the Ability (sv03.5-152/153/154, me03-068/069).
FOSSIL_HP_SHIFT, HP_GAP = (45, 44), 24
FOSSIL_GAP, FOSSIL_NAME_TO_BODY = 67, 42      # the effect's first baseline is the Technical Machines'
FOSSIL_TRACK = (-1.04, -1.4)
FOSSIL_ABILITY_SIZE = 25.5    # the Ability's text a little larger than on the Pokemon cards (their breaks: 24.9 breaks
                              # me03-069 a word late)

# Old rule sentences the Gen 9 pill prints instead (HeartGold SoulSilver and older).
# The Tool's attach sentence goes only when it restricts nothing (a name may hold a full stop: "EXP.ALL"); the
# Knocked Out reminder that follows it is the rule a Knocked Out Pokemon's cards go by anyway. Sentence variants
# tallied on the TCGdex text of 788 Trainers, Base Set to Sword & Shield.
OLD_RULES = [
    r"You can play only (?:one|1) Supporter card each turn\.",
    r"When you play this card, put it next to your Active Pokémon\.",
    r"When your turn ends, discard this card\.",
    r"This card stays in play (?:when you play it|after being played)\.",
    r"Discard this card if another Stadium card comes into play\.",
    r"If another card with the same name is in play, you can[’']t play this card\.",
    r"Attach (?:[^.]|\.(?=[A-Z]))+? to 1 of your Pokémon that doesn[’']t (?:already )?have a Pokémon Tool attached"
    r"(?: to it)?\.",
    r"If that Pokémon is Knocked Out, discard this card\.",
    # the rule box of Black & White and XY (TCGdex keeps it on promos and reprints: bw11-RC20, xyp-XY203)
    r"You may play only (?:one|1) Supporter card during your turn \(before your attack\)\.",
    r"You may play as many Item cards as you like during your turn \(before your attack\)\.",
]
# A Supporter that stays beside the Active Pokemon keeps the sentences that put it there and take it away when the
# rest of its text depends on them (dp6-130 Buck's Training: "As long as Buck's Training is next to ...").
PLACEMENT_RULES = OLD_RULES[1:3]
PLACED = r"\bis next to your Active Pokémon"
# Text that makes a Trainer a Pokemon Tool: the attach rule, or an attack the card lends (the e-Card Technical
# Machines, Cubes and Plates; TCGdex types none of those, and Curse Powder ex10-80 and the ex4 Technical Machines
# "Item" while their scans print "Pokémon Tool"). TCGdex misspells "may" on ecard2-121.
TOOL_RULES = (r"have a Pokémon Tool attached",
              r"Attach this card to 1 of your [^.]*\. That Pokémon m(?:a)?y use this card’s attack")
# Kind of a Trainer TCGdex gives no type (the first sets print "Trainer" on all of them): the rule its text holds.
UNTYPED_KINDS = (("Stadium", r"another Stadium card comes into play"), ("Supporter", r"Supporter card each turn"),
                 *(("Tool", rule) for rule in TOOL_RULES))
# Words TCGdex mistypes against the print (ecard2-121 Fighting Cube 01 prints "may"; ecard3-129 names itself).
TCGDEX_TYPOS = (("That Pokémon my use", "That Pokémon may use"), ("Miracle Plate α", "Miracle Sphere α"))
# xy4 prints "Team Flare Gear" as a label beside the name; TCGdex runs it into the name.
NAME_LABEL = r" Team Flare Gear$"
# A fossil's or doll's HP when TCGdex gives none but its text does (swsh10-157, xy4-102 "a 30 HP Colorless ...").
TEXT_HP = r"as if it were an? (\d+)[- ]HP\b"
TYPES = "Grass|Fire|Water|Lightning|Psychic|Fighting|Darkness|Metal|Dragon|Colorless"
CODE = {name: code for code, name in ts.ENERGY_CODES.items()}


# ---------------------------------------------------------------- card data
def era(card_id):
    """'modern' (SV / ME), an oldwindow layout, an ERA_ART era, or 'old' (no known window)."""
    set_id = card_id.split("-", 1)[0]
    if modern.is_modern(card_id):
        return "modern"
    layout = oldwindow.layout_of(card_id)      # before the eras: XY Evolutions prints Base Set's layout
    if layout:
        return layout
    for name, prefixes in ERAS.items():
        if set_id.startswith(prefixes):
            return name
    return "old"


def printed(text):
    """edit.printed (curly apostrophes, Pokémon), and curly double quotes as the prints set them (dp2-115)."""
    return re.sub(r'"([^"\n]*)"', r"“\1”", edit.printed(text))


def symbols(text):
    """Spelled-out types -> {G} codes where the newer cards print a symbol: before Pokémon or Energy (a basic or Evolved
    between: 'Fire and Lightning basic Energy', ecard3-129's scan prints symbols), in a list of types ending in one of
    those, before 'less' / 'more' (Retreat Costs), a Pokémon's type ('Metal type',
    'its type is Colorless': sv05-118 prints 'it is {F} and {M} type'), and run together (TCGdex writes two symbols
    as 'ColorlessColorless'). A Special Energy's name keeps its word (Double Colorless Energy)."""
    code = lambda m: f"{{{CODE[m[1]]}}}"
    text = re.sub(rf"\b((?:{TYPES}){{2,}})\b", lambda m: "".join(f"{{{CODE[t]}}}" for t in re.findall(TYPES, m[1])),
                  text)
    joined = r"\s*(?:,\s*(?:or|and)?|or|and)\s*"              # "Fire, Lightning, or Fighting Pokémon" (ecard3-129)
    listed = rf"(?<!Double )\b({TYPES})(?=(?:{joined}(?:{TYPES}|\{{[A-Z]\}}))*\s+(?:(?:[Bb]asic|Evolved)\s+)?(?:Pokémon|Energy)\b)"
    text = re.sub(listed, code, text)
    text = re.sub(rf"\b({TYPES})(?=\s+(?:less|more|type)\b)", code, text)
    return re.sub(rf"(\btype (?:\(color\) )?is )({TYPES})\b", lambda m: m[1] + f"{{{CODE[m[2]]}}}", text)


def effect_paragraphs(raw):
    """TCGdex effect text -> printed paragraphs: curly apostrophes and quotes, Pokémon, old rule sentences out,
    symbols, a choice's options as bullets."""
    text = printed(raw or "")
    for wrong, right in TCGDEX_TYPOS:
        text = text.replace(wrong, right)
    rules = [r for r in OLD_RULES if not (r in PLACEMENT_RULES and re.search(PLACED, text))]
    out = []
    for para in re.split(r"\n+", text):
        para = " ".join(para.split())
        for rule in rules:
            para = re.sub(rule, "", para)
        para = symbols(" ".join(para.split()))
        if not para:
            continue
        if out and out[-1].endswith(":") and not para.startswith("•"):
            # the options after "Choose 1 or more:" run together in TCGdex (swsh9-148 "deck.Shuffle"); printed as
            # bullets, as the cards that TCGdex gives them for (swsh1-171)
            out += [f"• {option}" for option in re.split(r"(?<=\.)(?=[A-Z])", para)]
            continue
        out.append(para)
    return out


def untyped_kind(card_id, raw):
    for kind, rule in UNTYPED_KINDS:
        if re.search(rule, printed(raw or "")):
            return kind
    warnings.warn(f"{card_id}: TCGdex gives no Trainer type and its text holds no rule: set as an Item")
    return "Item"


def kind_of(card_id, trainer_type, raw):
    """The Gen 9 frame of a Trainer: its TCGdex type's, but a Tool's when the text of one TCGdex types Item, or gives
    no type, has a Tool's rule."""
    if trainer_type == "Item" and any(re.search(rule, printed(raw)) for rule in TOOL_RULES):
        return "Tool"
    return KINDS[trainer_type] if trainer_type else untyped_kind(card_id, raw)


def with_hp(text, hp):
    """A fossil's or doll's HP in its 'as if it were a Basic Pokémon', worded as Gen 9 prints it ('a 60-HP Basic {C}
    Pokémon', sv03.5-152): xy4-102's 'a 30 HP {C} Basic Pokémon' too."""
    if not hp:
        return text

    def worded(m):
        symbol = m[1] or m[2]
        return f"as if it were a {hp}-HP Basic {symbol + ' ' if symbol else ''}Pokémon"
    return re.sub(r"as if it were an? (?:\d+[- ]HP )?(?:(\{[A-Z]\}) )?Basic (?:(\{[A-Z]\}) )?Pokémon", worded, text,
                  count=1)


def reprint_effect(card_id, name):
    """TCGdex leaves a few prints' text out (np-26 Tropical Wind): another print's of the same Trainer, from the same
    era first (the wording changes over the eras)."""
    briefs = [b for b in tcg.search("name", name) if b.get("id") and b["id"] != card_id]
    for brief in sorted(briefs, key=lambda b: (era(b["id"]) != era(card_id), b["id"])):
        other = tcg.card(brief["id"])
        if other.get("category") == "Trainer" and (other.get("effect") or "").strip():
            warnings.warn(f"{card_id}: TCGdex has no text for {name!r}: {brief['id']}'s used")
            return other["effect"]
    warnings.warn(f"{card_id}: TCGdex has no text for {name!r}, nor any other print of it: the panel stays empty")
    return ""


def split_name(name, kind):
    """'Boss's Orders (Giovanni)' -> ("Boss's Orders", "Giovanni"): Supporters print the character at the right."""
    m = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", name)
    if kind == "Supporter" and m:
        return m[1], m[2]
    return name, ""


def read_card(card_id):
    c = tcg.card(card_id)
    name = c.get("name", "")
    if c.get("category") != "Trainer":
        raise SystemExit(f"{card_id} is {name!r} ({c.get('category')}): not a Trainer card")
    if c.get("trainerType") not in KINDS:
        raise SystemExit(f"{card_id} is {name!r}: Trainer type {c.get('trainerType')!r} has no Gen 9 frame")
    set_id = c["set"]["id"]
    s = tcg.set_info(set_id)
    if (s.get("serie") or {}).get("id") == POCKET_SERIES:
        raise SystemExit(f"{card_id} is {name!r}, a Pokemon TCG Pocket card: there is no printed card to restyle")
    if set_id == CLASSIC_SET:
        raise SystemExit(f"{card_id} is {name!r}, a Classic Collection reprint: restyle the card it reprints")
    rarity = c.get("rarity") or ""
    if any(word in rarity.lower() for word in NOT_REGULAR):
        raise SystemExit(f"{card_id} is a {rarity} print ({name!r}): only regular Trainers, with an art window")
    if card_id in ACE_SPEC:
        raise SystemExit(f"{card_id} is {name!r}, an ACE SPEC card: its rule has no place on the Gen 9 frames")
    if PRISM in name or "Prism Star" in name:
        raise SystemExit(f"{card_id} is {name!r}: Prism Star cards have a rule Gen 9 prints no frame for")
    if name.endswith(HYPER_GEAR):
        raise SystemExit(f"{card_id} is {name!r}, attached to the opponent's Pokemon-EX: a rule the Gen 9 Tool "
                         "frame contradicts")
    raw = c.get("effect") or ""
    if TWO_WAY in raw:
        raise SystemExit(f"{card_id} is {name!r}, played either way up: its art holds the other way's text")
    if GX_ATTACK in raw:
        raise SystemExit(f"{card_id} is {name!r}, which gives a GX attack: the GX rule has no place on the Gen 9 "
                         "frames")
    if era(card_id) == "old":
        raise SystemExit(f"{card_id} is {name!r}: no known Trainer layout for set {set_id!r}, so no art window")
    if not raw.strip():
        raw = reprint_effect(card_id, name)
    kind = kind_of(card_id, c.get("trainerType"), raw)
    base, character = split_name(printed(re.sub(NAME_LABEL, "", name)), kind)
    attacks = [{"cost": list(a.get("cost") or []), "name": printed(a.get("name")),
                "damage": str(a.get("damage") or ""), "effect": " ".join(effect_paragraphs(a.get("effect")))}
               for a in c.get("attacks") or []]
    abilities = [{"name": printed(a["name"]), "effect": " ".join(effect_paragraphs(a["effect"]))}
                 for a in c.get("abilities") or [] if a.get("name") and a.get("effect")]
    for a in c.get("abilities") or []:
        if not (a.get("name") and a.get("effect")):
            warnings.warn(f"{card_id}: TCGdex lists a {a.get('type') or 'Ability'} without its name or text: left out")
    in_text = re.search(TEXT_HP, raw)
    hp = str(c.get("hp") or (in_text[1] if in_text else ""))
    official = (s.get("cardCount") or {}).get("official")
    return {
        "id": card_id, "kind": kind, "printed_kind": kind, "prefix": "", "name": base, "character": character,
        "hp": hp, "effect": with_hp("\n".join(effect_paragraphs(raw)), hp), "abilities": abilities,
        "attacks": attacks,
        "illustrator": c.get("illustrator") or "", "reg_mark": c.get("regulationMark") or "",
        "set_code": mecompose.set_code(set_id, s),
        "number": mecompose.card_number(c.get("localId", ""), official, s),
        "year": (s.get("releaseDate") or "")[:4], "rarity": mecompose.rarity_symbol(rarity), "era": era(card_id),
    }


# ---------------------------------------------------------------- art
def window(card, scan):
    """Where the old art is on this card (x0, y0, x1, y1)."""
    name = card["era"]
    if name == "modern":
        return ART
    if name in ERA_ART:
        return ERA_TOOL_ART.get(name, ERA_ART[name]) if card["printed_kind"] == "Tool" else ERA_ART[name]
    return oldwindow.window(card["id"], scan[..., :3])          # read_card refused the sets of no known layout


def old_art(card):
    """RGBA 733x1024: the old art, its badges painted out, scaled to cover the Gen 9 window (centred, the overhang
    cropped); cached."""
    out = cache.path("art", f"trainer-v{ART_VERSION}-{card['id']}.png")
    if out.exists():
        return load_rgba(out)
    scan = tcg.scan(card["id"])
    art = np.zeros_like(scan)
    x0, y0, x1, y1 = ART
    if card["era"] == "modern":
        art[y0:y1, x0:x1] = scan[y0:y1, x0:x1]
    else:
        if card["era"] in ERA_ART:
            scan, _ = badges.erase(scan, card["era"], ERA_ART[card["era"]])
        ox0, oy0, ox1, oy1 = window(card, scan)
        scale = max((x1 - x0) / (ox1 - ox0), (y1 - y0) / (oy1 - oy0))
        # the old window's middle lands on the new one's
        cx, cy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        m = np.array([[scale, 0, (x0 + x1) / 2 - cx * scale], [0, scale, (y0 + y1) / 2 - cy * scale]], np.float32)
        warped = cv2.warpAffine(scan, m, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        art[y0:y1, x0:x1] = warped[y0:y1, x0:x1]
    art[..., 3] = 255
    save_rgba(art, out)
    return art


# ---------------------------------------------------------------- text
def name_and_character(canvas, card):
    """Name, a Supporter's character and a fossil's HP. -> the name's horizontal squeeze."""
    room = NAME_RIGHT
    if card["character"] and card["kind"] == "Supporter":     # only Supporters print one
        width = ts.text_width(card["character"], CHARACTER_FACE, CHARACTER_SIZE)
        squeeze = min(1.0, CHARACTER_MAX_W / max(width, 1))
        if squeeze < 1:
            warnings.warn(f"character {card['character']!r} narrowed to {squeeze:.0%} to leave the name room")
        room = CHARACTER_RIGHT - width * squeeze - CHARACTER_GAP
        with layers.kind("character"):
            ts.stamp(canvas, ts.coverage([dict(text=card["character"], face=CHARACTER_FACE, size=CHARACTER_SIZE,
                                               x=CHARACTER_RIGHT, base=CHARACTER_BASE, anchor="r", xscale=squeeze)]),
                     CHARACTER_INK)
    if card.get("hp"):
        hp = _shifted(compose.hp_coverage(card["hp"]), *FOSSIL_HP_SHIFT)
        room = min(room, int(np.nonzero(hp.max(0) > 0.1)[0][0]) - HP_GAP)
        with layers.kind("hp"):
            ts.stamp(canvas, hp, np.array(frame.DARK_INK, np.float32))
    tm = card["name"].startswith(TM_PREFIX)
    size, widest, base = (TM_NAME_SIZE, TM_NAME_XSCALE, TM_NAME_BASE) if tm else (NAME_SIZE, NAME_XSCALE, NAME_BASE)
    if tm:
        room = min(room, TM_NAME_RIGHT)
    width = ts.text_width(card["name"], ts.BOLD, size)
    xscale = min(widest, (room - NAME_X) / max(width, 1))
    if xscale < NAME_MIN_XSCALE:
        # narrower than any print sets a name: smaller instead
        size, xscale = size * xscale / NAME_MIN_XSCALE, NAME_MIN_XSCALE
        warnings.warn(f"name {card['name']!r} set at {size:.0f} px to fit the bar")
    with layers.kind("name"):
        ts.stamp(canvas, ts.coverage([dict(text=card["name"], face=ts.BOLD, size=size, x=NAME_X, base=base,
                                           xscale=xscale)]), np.array(frame.DARK_INK, np.float32))
    return xscale


def _shifted(coverage, dx, dy):
    out = np.zeros_like(coverage)
    out[dy:, dx:] = coverage[:H - dy, :W - dx]
    return out


def snap(base):
    return GRID_TOP - GRID_STEP * round((GRID_TOP - base) / GRID_STEP)


def lay_paragraphs(paragraphs, k, blank=True, tracking=(BODY_TRACK, BODY_MIN_TRACK)):
    """-> ([(row, runs, tracking, symbol types, indent, bullet)], rows) at body scale k; a row is a line or, when
    blank, the blank between paragraphs (not between a choice's options). tracking: (natural, tightest)."""
    size = BODY_SIZE * k
    out, row = [], 0
    for i, para in enumerate(paragraphs):
        option = para.startswith(BULLET)
        if i and blank and not (option and paragraphs[i - 1].startswith(BULLET)):
            row += 1
        indent = BULLET_INDENT * k if option else 0
        text, types = ts.energy_text(ts.glue_numbers(para[len(BULLET):].lstrip() if option else para))
        marked = iter(types)
        for n, (runs, track) in enumerate(ts.compose_paragraph(text, size, BODY_FULL_W - indent, *tracking)):
            count = sum(t.count(ts.ENERGY_MARK) for t, _ in runs)
            out.append((row, runs, track, [next(marked) for _ in range(count)], indent, option and n == 0))
            row += 1
    return out, row


def _drop(lines):
    return SYMBOL_DROP if lines and lines[0][0] == 0 and lines[0][3] else 0


def draw_lines(canvas, lines, first, k):
    size, pitch = BODY_SIZE * k, BODY_PITCH * k
    items, drawn = [], np.zeros((H, W), np.float32)
    for row, runs, tracking, types, indent, bullet in lines:
        base, marks = first + row * pitch, []
        items += ts.line_items(runs, tracking, size, BODY_X + indent, base, marks)
        if bullet:
            items.append(dict(text=BULLET, face=BODY_FACE, size=size, x=BODY_X, base=base))
        for t, cx in zip(types, marks):
            np.maximum(drawn, compose.energy_symbol(t, cx, base, size), out=drawn)
    with layers.kind("text"):
        ts.stamp(canvas, np.maximum(ts.coverage(items), drawn), np.array(frame.DARK_INK, np.float32))


def body(canvas, card):
    """Effect paragraphs (then a fossil's Ability or a Technical Machine's attacks). -> the effect text's scale (1 =
    as printed)."""
    paragraphs = [p for p in card["effect"].split("\n") if p.strip()]
    if card["abilities"]:
        return blocks_body(canvas, card, paragraphs, TM_FIRST_BASE, FOSSIL_GAP, FOSSIL_NAME_TO_BODY, blank=False,
                           tracking=FOSSIL_TRACK, block_size=FOSSIL_ABILITY_SIZE)
    if card["attacks"]:
        return blocks_body(canvas, card, paragraphs, TM_FIRST_BASE, TM_GAP, TM_NAME_TO_BODY)
    for k in SHRINK_STEPS:
        lines, rows = lay_paragraphs(paragraphs, k)
        first = max(snap(BLOCK_CENTRE - (rows - 1) * ROW_HALF * k), BODY_TOP_MIN) + _drop(lines)
        if first + (rows - 1) * BODY_PITCH * k <= BODY_BOTTOM_MAX:
            break
    else:
        warnings.warn(f"{card['id']}: the effect text runs taller than the panel holds")
    draw_lines(canvas, lines, first, k)
    return k


def blocks_body(canvas, card, paragraphs, first_base, gap, name_to_body, blank=True,
                tracking=(BODY_TRACK, BODY_MIN_TRACK), block_size=compose.BODY_SIZE):
    """The effect from a fixed first baseline, then Abilities and attacks set as on the Pokemon cards (badge or
    energy costs, name, damage; text at their size), the first name's cap top gap below the effect's last baseline."""
    blocks = len(card["abilities"]) + len(card["attacks"])
    for k in SHRINK_STEPS:
        lines, rows = lay_paragraphs(paragraphs, k, blank, tracking)
        first = first_base + _drop(lines)
        last = first + (rows - 1) * BODY_PITCH * k if lines else first - gap
        if last + gap + BLOCK_ROOM * blocks <= PANEL_BOTTOM:
            break
    else:
        warnings.warn(f"{card['id']}: the effect, abilities and attacks run taller than the panel holds")
    draw_lines(canvas, lines, first, k)
    top = last + gap

    def layout(laid, pitch):
        tops, t = [], top
        for _, _, block_lines in laid:
            tops.append(t)
            t += (name_to_body + pitch * (len(block_lines) - 1) + mecompose.LINE_CAP if block_lines
                  else mecompose.ROW_H) + mecompose.TIGHT_GAP
        return tops, t - mecompose.TIGHT_GAP <= PANEL_BOTTOM
    compose.text_block(canvas, {"id": card["id"], "abilities": card["abilities"], "attacks": card["attacks"]},
                       np.array(frame.DARK_INK, np.float32), None, layout=layout, name_to_body=name_to_body,
                       fit_tracking=mecompose.BODY_FIT_TRACK, cost_r=mecompose.COST_R, cost_dy=TM_COST_DY,
                       body_size=block_size, dmg_limit=BODY_X + BODY_FULL_W)
    return k


# ---------------------------------------------------------------- card
def warn_missing_glyphs(card):
    """Gill Sans has no Greek (ecard3-129 Miracle Sphere α): name the characters that would print as a box."""
    blocks = card["abilities"] + card["attacks"]
    texts = ((card["name"], ts.BOLD), (card["character"], CHARACTER_FACE), (card["effect"], BODY_FACE),
             *((b.get("name") or "", compose.ATK_FACE) for b in blocks),
             *((b.get("effect") or "", BODY_FACE) for b in blocks))
    missing = sorted({ch for text, face in texts for ch in ts.missing_glyphs(text, face)})
    if missing:
        warnings.warn(f"{card['id']}: Gill Sans has no {' '.join(missing)}: it prints as an empty box (edit the text)")


def compose_card(card_id, card=None):
    """-> (RGBA float32 733x1024, report dict). card: read_card's dict, possibly edited (default: read it)."""
    card = card or read_card(card_id)
    warn_missing_glyphs(card)
    template = trainerframe.template(card["kind"])
    art = old_art(card)
    a = template[..., 3:4] / 255
    if card["era"] == "modern":
        a = (a >= 1).astype(np.float32)        # the print's window already holds its rim's shading
    canvas = np.zeros((H, W, 4), np.float32)
    canvas[..., :3] = art[..., :3] * (1 - a) + template[..., :3] * a
    canvas[..., 3] = trainerframe.corners()
    with layers.kind("frame"):
        layers.record(a[..., 0])
    xscale = name_and_character(canvas, card)
    scale = body(canvas, card)
    compose.footer(canvas, card, False, rarity=card["rarity"], shift=FOOTER_SHIFT, keyline=False)
    return canvas, {"card": card, "name_xscale": round(xscale, 3), "body_scale": scale}
