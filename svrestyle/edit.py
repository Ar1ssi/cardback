"""The part of a card the GUI lets you change, and the checks on it.

A card dict comes from compose.read_card (ex layout), mecompose.read_card (Mega Evolution layout),
trainer.read_card (Trainers: they have a "kind" and no retreat) or energy.read_card (Energy: an "energy" of "basic" or
"special"). editable() picks the fields shown in the editor;
apply() writes edited values back onto a copy after checking them, so the composers only ever see well-formed data.
Regular Pokemon (family "regular": one Prize, no rule box) can change type, and so can the cards whose reader says so
("retype": a Pokemon V restyled as ex, a Mega ex print megaretype recolours); other ex cards keep theirs.
Effect text (abilities, attacks, Trainer text) also gets the TCG's fixed terms in their printed capitals
(wording.fix_terms); names and flavour text are kept as typed.
"""
import copy
import re

from . import wording

ENERGY = ("Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal", "Fairy", "Dragon",
          "Colorless")
MAX_ABILITIES, MAX_ATTACKS, MAX_COST, MAX_RETREAT = 2, 4, 5, 5
STAGES = (0, 1, 2, 3)                                # Basic, Stage 1, Stage 2, Mega (custom: stage.MEGA)
MAX_NAME, MAX_EFFECT, MAX_FLAVOR = 40, 400, 220
# Trainers: their kind (the Gen 9 frame), and their text, paragraphs a line apart (the longest old ones, fossils,
# run to 500 characters).
TRAINER_KINDS = ("Item", "Supporter", "Stadium", "Tool")
MAX_TRAINER_TEXT, MAX_PARAGRAPHS = 800, 6
ACE_FIXED = ("kind", "character", "hp", "abilities", "attacks")      # what an ACE SPEC card cannot change (ace.py)
TRAINER_SCALAR = {"name": 60, "character": MAX_NAME, "illustrator": MAX_NAME}
# Energy: a Special Energy's name, text, illustrator, the Energy it provides (the name bar's icons) and its frame.
ENERGY_SCALAR = {"name": 60, "illustrator": MAX_NAME}
ENERGY_FRAMES = ("regular", "full", "typed")
# The typed frame: the Mega Evolution typed Special Energy (Telepathic Psychic Energy and its like, energy.py) of the one
# type the card provides, graded as they are.
TYPED_TYPES = ("Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal", "Fairy")
TYPED_RARITY = "Rare"
# An older Basic Energy: the Mega Evolution Basic Energy of its type, or its type's gold Hyper Rare one (energy.py).
BASIC_FRAMES, GOLD_RARITY = ("basic", "gold"), "Hyper rare"
MAX_PROVIDED = 3

DAMAGE = re.compile(r"(\d{1,4})?([+×x-])?")        # 120, 30+, 50×, 10-, or empty (no damage printed)
WEAKNESS = re.compile(r"[×x+]\d{1,3}")              # ×2 (current), +20 (older cards)
RESISTANCE = re.compile(r"[-−]\d{1,3}")              # -30
# What cards print, offered as choices in the editor (any value the patterns accept still works).
WEAKNESS_VALUES = ("×2", "+10", "+20", "+30", "+40")
RESISTANCE_VALUES = ("-10", "-20", "-30", "-40")
CARD_ID = re.compile(r"[a-z]{2,6}[0-9.]*[a-z]?-[A-Za-z0-9]{1,10}")   # swsh6-99, sv03.5-065, sma-SV33, swshp-SWSH055

SCALAR = {"prefix": MAX_NAME, "name": MAX_NAME, "hp": 4, "illustrator": MAX_NAME, "evolves_from": MAX_NAME,
          "flavor": MAX_FLAVOR}


def valid_card_id(card_id):
    return isinstance(card_id, str) and CARD_ID.fullmatch(card_id) is not None


POKEMON = re.compile(r"\b(pok)e(mon)", re.IGNORECASE)   # Pokemon, POKEMON, Pokemon's, Pokemon-ex


def printed(text):
    """Cards print a curly apostrophe and an accented Pokémon."""
    text = POKEMON.sub(lambda m: m[1] + ("É" if m[0][3] == "E" else "é") + m[2], text or "")
    return text.replace("'", "’")


def is_trainer(card):
    return "kind" in card


def is_energy(card):
    return "energy" in card


def retypes(card):
    """Whether the card's type can be changed: regular Pokemon, a Pokemon V restyled as ex (set on its new type's
    frame) and the Mega ex prints megaretype recolours ("retype"); a Scarlet & Violet ex's frame is its own."""
    return card.get("family") == "regular" or bool(card.get("retype"))


def editable(card):
    """The editor's view of a card (plain JSON)."""
    if is_energy(card):
        return _energy_editable(card)
    if is_trainer(card):
        return _trainer_editable(card)
    out = {k: card.get(k, "") for k in SCALAR if k in card}
    if retypes(card):
        out["type"] = card["type"]
    out["abilities"], out["attacks"] = _blocks_view(card)
    for side in ("weakness", "resistance"):
        v = card.get(side)
        out[side] = {"type": v["type"], "value": v.get("value", "")} if v else None
    out["retreat"] = card["retreat"]
    out["stage"] = card.get("stage", 0)
    out["portrait"] = card.get("portrait", "")
    out["mega_rule"] = bool(card.get("mega_rule"))
    return out


def _blocks_view(card):
    return ([{"name": a.get("name", ""), "effect": a.get("effect", "")} for a in card["abilities"]],
            [{"cost": list(a.get("cost") or []), "name": a.get("name", ""), "damage": str(a.get("damage") or ""),
              "effect": a.get("effect") or ""} for a in card["attacks"]])


def _trainer_editable(card):
    out = {k: card.get(k, "") for k in TRAINER_SCALAR}
    out.update(kind=card["kind"], hp=card.get("hp", ""), effect=card.get("effect", ""))
    out["abilities"], out["attacks"] = _blocks_view(card)
    return out


def _energy_fixed(card):
    """Energy with nothing to edit: a Basic Energy (the type's own, only its footer is set) and Energy that is in the
    look already (frame "print")."""
    return card["energy"] == "basic" or card["frame"] == "print"


def _energy_editable(card):
    if _energy_fixed(card):
        out = {"energy": card["energy"], "frame": card["frame"]}
        if "type" in card:
            out["type"] = card["type"]
        return out
    out = {k: card.get(k, "") for k in ENERGY_SCALAR}
    out.update(energy="special", effect=card.get("effect", ""), cost=list(card.get("cost") or []),
               frame=card["frame"])
    return out


def _text(value, field, limit, required=False):
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected text")
    value = " ".join(value.split())
    if len(value) > limit:
        raise ValueError(f"{field}: at most {limit} characters")
    if required and not value:
        raise ValueError(f"{field}: required")
    return printed(value)


def _energy(value, field):
    if value not in ENERGY:
        raise ValueError(f"{field}: unknown energy type {value!r} (use one of {', '.join(ENERGY)})")
    return value


def _list(value, field, limit):
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected a list")
    if len(value) > limit:
        raise ValueError(f"{field}: at most {limit}")
    return value


def _ability(a, i):
    if not isinstance(a, dict):
        raise ValueError(f"ability {i + 1}: expected an object")
    return {"name": _text(a.get("name", ""), f"ability {i + 1} name", MAX_NAME, required=True),
            "effect": wording.fix_terms(_text(a.get("effect", ""), f"ability {i + 1} text", MAX_EFFECT, required=True))}


def _attack(a, i):
    if not isinstance(a, dict):
        raise ValueError(f"attack {i + 1}: expected an object")
    field = f"attack {i + 1}"
    cost = [_energy(t, f"{field} cost") for t in _list(a.get("cost", []), f"{field} cost", MAX_COST)]
    damage = str(a.get("damage", "") or "").strip().replace("x", "×").replace("−", "-")
    m = DAMAGE.fullmatch(damage)
    if not m or (m.group(2) and not m.group(1)):
        raise ValueError(f"{field} damage: a number with an optional +, × or - after it (e.g. 120, 30+, 50×)")
    return {"cost": cost, "name": _text(a.get("name", ""), f"{field} name", MAX_NAME, required=True),
            "damage": damage, "effect": wording.fix_terms(_text(a.get("effect", "") or "", f"{field} text", MAX_EFFECT))}


def _side(v, field, pattern, example):
    if v in (None, "", {}):
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{field}: expected an object or none")
    value = str(v.get("value", "")).strip().replace("x", "×").replace("−", "-")
    if not pattern.fullmatch(value):
        raise ValueError(f"{field} value: like {example}")
    return {"type": _energy(v.get("type"), f"{field} type"), "value": value}


def _paragraphs(value, field):
    """Text of paragraphs a line apart (a blank line or a line break): each one's spacing tidied, empty ones out."""
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected text")
    if len(value) > MAX_TRAINER_TEXT:
        raise ValueError(f"{field}: at most {MAX_TRAINER_TEXT} characters")
    paragraphs = [wording.fix_terms(printed(" ".join(p.split()))) for p in value.splitlines() if p.strip()]
    if len(paragraphs) > MAX_PARAGRAPHS:
        raise ValueError(f"{field}: at most {MAX_PARAGRAPHS} paragraphs")
    return "\n".join(paragraphs)


def _apply_trainer(out, edits):
    card = copy.deepcopy(out)
    for key, limit in TRAINER_SCALAR.items():
        if key in edits:
            out[key] = _text(edits[key], key, limit, required=key == "name")
    if "kind" in edits:
        if edits["kind"] not in TRAINER_KINDS:
            raise ValueError(f"kind: one of {', '.join(TRAINER_KINDS)}")
        out["kind"] = edits["kind"]
    if "hp" in edits:
        out["hp"] = _text(edits["hp"], "hp", 4)
        if out["hp"] and not re.fullmatch(r"\d{1,3}0", out["hp"]):
            raise ValueError("hp: empty, or a multiple of 10 up to 9990 (fossils and dolls)")
    if "effect" in edits:
        out["effect"] = _paragraphs(edits["effect"], "text")
    if "abilities" in edits:
        out["abilities"] = [_ability(a, i) for i, a in enumerate(_list(edits["abilities"], "abilities",
                                                                        MAX_ABILITIES))]
    if "attacks" in edits:
        out["attacks"] = [_attack(a, i) for i, a in enumerate(_list(edits["attacks"], "attacks", MAX_ATTACKS))]
    if out.get("ace") and any(out[k] != card[k] for k in ACE_FIXED):
        raise ValueError("an ACE SPEC card keeps its own frame: only its name, text and illustrator can change")
    return out


def _apply_energy(out, edits):
    """Edits the editor sends back include the card's "energy" and a Basic Energy's "type", as editable() gave them."""
    if edits.get("energy", out["energy"]) != out["energy"]:
        raise ValueError("energy: a Basic Energy cannot become a Special Energy, or the other way round")
    if _energy_fixed(out):
        frame = edits.get("frame", out["frame"])
        if out["energy"] == "basic" and out["frame"] in BASIC_FRAMES and frame in BASIC_FRAMES:
            out["frame"] = frame                    # an older Basic Energy: the Mega Evolution one or the gold one
            if frame == "gold":                     # named and graded as the gold prints are (Basic Psychic Energy)
                out["rarity_name"] = GOLD_RARITY
                out["prefix"], out["name"] = "", f"Basic {out['type']} Energy"
            edits = {k: v for k, v in edits.items() if k != "frame"}
        if any(k not in ("energy", "type", "frame") or edits[k] != out[k] for k in edits):
            raise ValueError("this Energy has nothing to edit: a Basic Energy prints only its set code and number (on "
                             "the Mega Evolution or the gold frame), and Scarlet & Violet / Mega Evolution Energy keeps "
                             "its print")
        return out
    for key, limit in ENERGY_SCALAR.items():
        if key in edits:
            out[key] = _text(edits[key], key, limit, required=key == "name")
    if "effect" in edits:
        out["effect"] = _paragraphs(edits["effect"], "text")
    if "cost" in edits:
        out["cost"] = [_energy(t, "provides") for t in _list(edits["cost"], "provides", MAX_PROVIDED)]
    if "frame" in edits:
        if edits["frame"] not in ENERGY_FRAMES:
            raise ValueError(f"frame: one of {', '.join(ENERGY_FRAMES)}")
        out["frame"] = edits["frame"]
    if out["frame"] == "typed":
        provided = set(out.get("cost") or [])
        if len(provided) != 1 or not provided <= set(TYPED_TYPES):
            raise ValueError("frame typed: the Mega Evolution typed Special Energy shows one type: set it to provide "
                             f"one of {', '.join(TYPED_TYPES)}")
        out["rarity_name"], out["rarity"] = TYPED_RARITY, "rare"
    return out


def apply(card, edits):
    """A copy of card with the edits applied. Raises ValueError (with a message for the user) on bad input."""
    if not isinstance(edits, dict):
        raise ValueError("edits: expected an object")
    out = copy.deepcopy(card)
    if is_energy(card):
        return _apply_energy(out, edits)
    if is_trainer(card):
        return _apply_trainer(out, edits)
    for key, limit in SCALAR.items():
        if key in edits and key in card:
            out[key] = _text(edits[key], key, limit, required=key == "name")
    if "hp" in edits:
        if not re.fullmatch(r"\d{1,3}0", out["hp"]):
            raise ValueError("hp: a multiple of 10, up to 9990")
    if "type" in edits and edits["type"] != card["type"]:
        if not retypes(card):
            raise ValueError("type: this card keeps its type (regular Pokémon, Pokémon V restyled as ex and Mega "
                             "Evolution ex can change theirs)")
        out["type"] = _energy(edits["type"], "type")
    if "abilities" in edits:
        out["abilities"] = [_ability(a, i) for i, a in enumerate(_list(edits["abilities"], "abilities",
                                                                        MAX_ABILITIES))]
    if "attacks" in edits:
        out["attacks"] = [_attack(a, i) for i, a in enumerate(_list(edits["attacks"], "attacks", MAX_ATTACKS))]
    if "weakness" in edits:
        out["weakness"] = _side(edits["weakness"], "weakness", WEAKNESS, "×2 or +20")
    if "resistance" in edits:
        out["resistance"] = _side(edits["resistance"], "resistance", RESISTANCE, "-30")
    if "retreat" in edits:
        r = edits["retreat"]
        if not isinstance(r, int) or isinstance(r, bool) or not 0 <= r <= MAX_RETREAT:
            raise ValueError(f"retreat: a whole number from 0 to {MAX_RETREAT}")
        out["retreat"] = r
    if "stage" in edits:
        s = edits["stage"]
        if not isinstance(s, int) or isinstance(s, bool) or s not in STAGES:
            raise ValueError("stage: 0 (Basic), 1 (Stage 1), 2 (Stage 2) or 3 (Mega)")
        out["stage"] = s
    if "portrait" in edits:
        p = edits["portrait"]
        p = p.strip() if isinstance(p, str) else p
        if p and not valid_card_id(p):
            raise ValueError("portrait: a TCGdex card id (like sv03.5-064), or empty to find one by name")
        out["portrait"] = p or ""
    if "mega_rule" in edits:
        if not isinstance(edits["mega_rule"], bool):
            raise ValueError("mega rule: true or false")
        out["mega_rule"] = edits["mega_rule"]
    if out.get("stage") and not out.get("evolves_from"):
        raise ValueError("evolves from: a Stage 1, Stage 2 or Mega card names what it evolves from")
    return out


# ---------------------------------------------------------------- what changed
STAGE_NAMES = ("Basic", "Stage 1", "Stage 2", "Mega")
LETTER = {"Grass": "G", "Fire": "R", "Water": "W", "Lightning": "L", "Psychic": "P", "Fighting": "F",
          "Darkness": "D", "Metal": "M", "Fairy": "Y", "Dragon": "N", "Colorless": "C"}     # as typed in effect text


def _cost(cost):
    return "".join(f"{{{LETTER.get(t, '?')}}}" for t in cost) or "no cost"


def _quote(text):
    return f"“{text}”" if text else "none"


def _was(field, old, new):
    return f"{field}: {old} → {new}"


def _ability_diff(old, new):
    out = [_was("name", old["name"], new["name"])] if old["name"] != new["name"] else []
    if old["effect"] != new["effect"]:
        out.append(_was("text", _quote(old["effect"]), _quote(new["effect"])))
    return out


def _attack_diff(old, new):
    out = [_was("name", old["name"], new["name"])] if old["name"] != new["name"] else []
    if old["cost"] != new["cost"]:
        out.append(_was("cost", _cost(old["cost"]), _cost(new["cost"])))
    if old["damage"] != new["damage"]:
        out.append(_was("damage", old["damage"] or "none", new["damage"] or "none"))
    if old["effect"] != new["effect"]:
        out.append(_was("text", _quote(old["effect"]), _quote(new["effect"])))
    return out


def _attack_line(a):
    damage = f" {a['damage']}" if a["damage"] else ""
    return f"{_cost(a['cost'])} {a['name']}{damage}" + (f": {_quote(a['effect'])}" if a["effect"] else "")


def _blocks(kind, old, new, diff, line):
    """Abilities or attacks: paired by name, then what is left by position (a rename); the rest added or removed."""
    key = lambda b: b["name"].casefold()
    left, right, pairs = list(range(len(old))), list(range(len(new))), []
    for j in list(right):
        i = next((i for i in left if key(old[i]) == key(new[j])), None)
        if i is not None:
            pairs.append((i, j))
            left.remove(i)
            right.remove(j)
    while left and right:
        pairs.append((left.pop(0), right.pop(0)))
    pairs.sort(key=lambda p: p[1])
    out = []
    for i, j in pairs:
        d = diff(old[i], new[j])
        if d:
            out.append(f"{kind} {old[i]['name']}: " + "; ".join(d))
    out += [f"{kind} removed: {line(old[i])}" for i in left]
    out += [f"{kind} added: {line(new[j])}" for j in right]
    order = [i for i, _ in pairs]
    if order != sorted(order):
        out.append(f"{kind} order: " + ", ".join(new[j]["name"] for _, j in pairs))
    return out


def _trainer_changes(a, b):
    out = []
    for key, label in (("name", "Name"), ("character", "Character"), ("kind", "Trainer type"), ("hp", "HP")):
        if a.get(key) != b.get(key):
            out.append(_was(label, a.get(key) or "none", b.get(key) or "none"))
    if a["effect"] != b["effect"]:
        out.append(_was("Text", _quote(a["effect"].replace("\n", " / ")), _quote(b["effect"].replace("\n", " / "))))
    out += _blocks("Ability", a["abilities"], b["abilities"], _ability_diff,
                   lambda x: f"{x['name']}: {_quote(x['effect'])}")
    out += _blocks("Attack", a["attacks"], b["attacks"], _attack_diff, _attack_line)
    if a.get("illustrator") != b.get("illustrator"):
        out.append(_was("Illustrator", _quote(a.get("illustrator")), _quote(b.get("illustrator"))))
    return out


def _energy_changes(a, b):
    out = []
    if a.get("name") != b.get("name"):
        out.append(_was("Name", a.get("name"), b.get("name")))
    if a.get("effect") != b.get("effect"):
        out.append(_was("Text", _quote(a["effect"].replace("\n", " / ")), _quote(b["effect"].replace("\n", " / "))))
    if a.get("cost") != b.get("cost"):
        out.append(_was("Provides", _cost(a["cost"]), _cost(b["cost"])))
    if a.get("frame") != b.get("frame"):
        out.append(_was("Frame", a.get("frame"), b.get("frame")))
    if a.get("illustrator") != b.get("illustrator"):
        out.append(_was("Illustrator", _quote(a.get("illustrator")), _quote(b.get("illustrator"))))
    return out


def changes(before, after):
    """What differs between two card dicts, one line per change, in the order the card prints it."""
    a, b = editable(before), editable(after)
    if is_energy(before):
        return _energy_changes(a, b)
    if is_trainer(before):
        return _trainer_changes(a, b)
    out = []
    name = lambda c: f"{c.get('prefix', '')} {c.get('name', '')}".strip()
    if name(a) != name(b):
        out.append(_was("Name", name(a), name(b)))
    if a.get("hp") != b.get("hp"):
        out.append(_was("HP", a.get("hp"), b.get("hp")))
    if a.get("type") != b.get("type"):
        out.append(_was("Type", a.get("type"), b.get("type")))
    if a.get("stage", 0) != b.get("stage", 0):
        out.append(_was("Stage", STAGE_NAMES[a.get("stage", 0)], STAGE_NAMES[b.get("stage", 0)]))
    if b.get("stage") and a.get("evolves_from") != b.get("evolves_from"):
        out.append(_was("Evolves from", a.get("evolves_from") or "none", b.get("evolves_from")))
    if b.get("stage") and a.get("portrait") != b.get("portrait"):
        out.append(f"Portrait: from {b['portrait']}" if b.get("portrait") else "Portrait: found by name")
    if a.get("mega_rule") != b.get("mega_rule"):
        out.append("Mega Evolution rule (XY): " + ("added" if b.get("mega_rule") else "taken off"))
    out += _blocks("Ability", a["abilities"], b["abilities"], _ability_diff, lambda x: f"{x['name']}: {_quote(x['effect'])}")
    out += _blocks("Attack", a["attacks"], b["attacks"], _attack_diff, _attack_line)
    for side in ("weakness", "resistance"):
        old, new = a.get(side), b.get(side)
        if old != new:
            show = lambda v: f"{v['type']} {v['value']}" if v else "none"
            out.append(_was(side.capitalize(), show(old), show(new)))
    if a["retreat"] != b["retreat"]:
        out.append(_was("Retreat", a["retreat"], b["retreat"]))
    for key, label in (("illustrator", "Illustrator"), ("flavor", "Flavour text")):
        if a.get(key) != b.get(key):
            out.append(_was(label, _quote(a.get(key)), _quote(b.get(key))))
    return out
