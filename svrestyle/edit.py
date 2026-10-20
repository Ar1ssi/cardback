"""The part of a card the GUI lets you change, and the checks on it.

A card dict comes from compose.read_card (ex layout) or mecompose.read_card (Mega Evolution layout). editable()
picks the fields shown in the editor; apply() writes edited values back onto a copy after checking them, so the
composers only ever see well-formed data.
"""
import copy
import re

ENERGY = ("Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal", "Dragon", "Colorless")
MAX_ABILITIES, MAX_ATTACKS, MAX_COST, MAX_RETREAT = 2, 4, 5, 5
MAX_NAME, MAX_EFFECT, MAX_FLAVOR = 40, 400, 220

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


def editable(card):
    """The editor's view of a card (plain JSON)."""
    out = {k: card.get(k, "") for k in SCALAR if k in card}
    out["abilities"] = [{"name": a.get("name", ""), "effect": a.get("effect", "")} for a in card["abilities"]]
    out["attacks"] = [{"cost": list(a.get("cost") or []), "name": a.get("name", ""),
                       "damage": str(a.get("damage") or ""), "effect": a.get("effect") or ""}
                      for a in card["attacks"]]
    for side in ("weakness", "resistance"):
        v = card.get(side)
        out[side] = {"type": v["type"], "value": v.get("value", "")} if v else None
    out["retreat"] = card["retreat"]
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
            "effect": _text(a.get("effect", ""), f"ability {i + 1} text", MAX_EFFECT, required=True)}


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
            "damage": damage, "effect": _text(a.get("effect", "") or "", f"{field} text", MAX_EFFECT)}


def _side(v, field, pattern, example):
    if v in (None, "", {}):
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{field}: expected an object or none")
    value = str(v.get("value", "")).strip().replace("x", "×").replace("−", "-")
    if not pattern.fullmatch(value):
        raise ValueError(f"{field} value: like {example}")
    return {"type": _energy(v.get("type"), f"{field} type"), "value": value}


def apply(card, edits):
    """A copy of card with the edits applied. Raises ValueError (with a message for the user) on bad input."""
    if not isinstance(edits, dict):
        raise ValueError("edits: expected an object")
    out = copy.deepcopy(card)
    for key, limit in SCALAR.items():
        if key in edits and key in card:
            out[key] = _text(edits[key], key, limit, required=key == "name")
    if "hp" in edits:
        if not re.fullmatch(r"\d{1,3}0", out["hp"]):
            raise ValueError("hp: a multiple of 10, up to 9990")
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
    return out
