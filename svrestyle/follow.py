"""What goes with a Pokemon's type besides its frame: the Energy it pays with and the weakness and resistance its type's
cards print. The editor's "Follow the type" sets them for the card's new type (follow).

- Costs and effect text: the card's own type's Energy (costs, {X} in effect text) becomes the new type's. Dragon pays
  with other types' Energy: a card made Dragon keeps its costs, and a Dragon card made another type pays that type's
  for every coloured Energy; effect text keeps its symbols either way. Colorless costs are paid with any Energy: a
  Colorless card's costs and text stay as they are (a card made Colorless still pays Colorless for its own type's).
- Weakness and resistance: the ones most of the new type's cards print (rules): Mega ex take the Mega ex Double Rares'
  (megaretype.SOURCES), every other card the Mega Evolution regular cards' (meframe.POOL); the weakness keeps the
  card's own value (x2). Dragon prints none. Fairy, which neither era prints, takes Sun & Moon's: weak to Metal,
  resisting Darkness, at the new era's -30.
Each follow starts from the card as printed, so changing type again re-maps the printed costs, not the last follow's;
costs and text the user changed otherwise are mapped from the printed type as they stand.
"""
import functools
import re
from collections import Counter

from . import edit, meframe, megaretype, tcg

FAIRY = ({"type": "Metal", "value": "×2"}, {"type": "Darkness", "value": "-30"})
SYMBOL = re.compile(r"\{([A-Z])\}")


def _value(v):
    return (v or "").replace("−", "-").replace("x", "×")


@functools.lru_cache(maxsize=None)
def rules(mega):
    """{type: (weakness type or None, (resistance type, value) or None)} most of the type's cards print; mega: the
    Mega ex Double Rares', else the Mega Evolution regular cards'."""
    pool = megaretype.SOURCES if mega else meframe.POOL
    out = {}
    for t, ids in pool.items():
        weak, res = Counter(), Counter()
        for card_id in ids:
            c = tcg.card(card_id)
            w, r = (c.get("weaknesses") or [None])[0], (c.get("resistances") or [None])[0]
            weak[w and w["type"]] += 1
            res[r and (r["type"], _value(r.get("value")))] += 1
        out[t] = (weak.most_common(1)[0][0], res.most_common(1)[0][0])
    out["Fairy"] = (FAIRY[0]["type"], (FAIRY[1]["type"], FAIRY[1]["value"]))
    return out


def _mine(old_t):
    """Whether a cost of type t is paid with the card's own type's Energy (a Colorless card's: none, its costs take
    any Energy)."""
    if old_t == "Colorless":
        return lambda t: False
    return (lambda t: t != "Colorless") if old_t == "Dragon" else (lambda t: t == old_t)


def costs(cost, old_t, new_t):
    """A cost list following a type change old_t -> new_t."""
    mine = _mine(old_t)
    return [new_t if mine(t) and new_t != "Dragon" else t for t in cost]


def _followed(printed, now, old_t):
    """Whether cost list now is printed's after some follow: the same, its own type's Energy all one other type."""
    if len(printed) != len(now):
        return False
    mine = _mine(old_t)
    taken = {n for p, n in zip(printed, now) if mine(p)}
    return all(p == n for p, n in zip(printed, now) if not mine(p)) and len(taken) <= 1


def text(effect, old_t, new_t):
    """Effect text following a type change: its {X} of the old type are the new type's (not to or from Dragon)."""
    if "Dragon" in (old_t, new_t) or old_t in (new_t, "Colorless"):
        return effect
    return (effect or "").replace(f"{{{edit.LETTER[old_t]}}}", f"{{{edit.LETTER[new_t]}}}")


def _text_followed(printed, now, old_t):
    """Whether effect text now is printed's with its own type's symbols all one other type's."""
    a, b = SYMBOL.split(printed or ""), SYMBOL.split(now or "")
    if len(a) != len(b) or a[::2] != b[::2]:
        return False
    own = edit.LETTER.get(old_t) if old_t != "Colorless" else None
    taken = {y for x, y in zip(a[1::2], b[1::2]) if x == own}
    return all(x == y for x, y in zip(a[1::2], b[1::2]) if x != own) and len(taken) <= 1


def follow(original, card):
    """card (edited from original, the card as printed) with its costs, effect text, weakness and resistance
    following its type (card["type"]), as edits (edit.editable's shape for abilities, attacks, weakness and
    resistance)."""
    old_t, new_t = original["type"], card["type"]
    mega = original.get("family") == "ex" and original["name"].startswith("Mega ")

    def block(i, b, printed_blocks):
        p = printed_blocks[i] if i < len(printed_blocks) and printed_blocks[i]["name"] == b["name"] else None
        out = dict(b)
        if p is not None and _text_followed(p.get("effect"), b.get("effect"), old_t):
            out["effect"] = text(p.get("effect") or "", old_t, new_t)
        else:
            out["effect"] = text(b.get("effect") or "", old_t, new_t)
        if "cost" in b:
            if p is not None and _followed(p.get("cost") or [], b.get("cost") or [], old_t):
                out["cost"] = costs(p.get("cost") or [], old_t, new_t)
            else:
                out["cost"] = costs(b.get("cost") or [], old_t, new_t)
        return out

    view = edit.editable(card)
    printed = edit.editable(original)
    weak_t, res = rules(mega).get(new_t, (None, None))
    value = (card.get("weakness") or {}).get("value") or "×2"
    return {
        "abilities": [block(i, b, printed["abilities"]) for i, b in enumerate(view["abilities"])],
        "attacks": [block(i, b, printed["attacks"]) for i, b in enumerate(view["attacks"])],
        "weakness": {"type": weak_t, "value": value} if weak_t else None,
        "resistance": {"type": res[0], "value": res[1]} if res else None,
    }
