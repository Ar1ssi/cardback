"""Custom sets for ptcg-sim: every card of the editor's list as one set the simulator plays.

The zip holds one folder, ready to drop into the simulator's client/src/assets/custom-sets/:

    <set id>/set.json            the set and its cards, in TCGdex's record shape (the simulator's proxy serves
                                 them as TCGdex records, so its deck builder, enrichment and rules engine read
                                 them like any published card)
    <set id>/<nnn>/high.webp     the rendered card, 733x1024
    <set id>/<nnn>/low.webp      the same at 245x342 (the simulator's grid tiles)
    <set id>/CHANGELOG.md        every card's changes from its printed card

Set code: 2-5 capital letters or digits, as a decklist prints it ("4 Tyranitar HBW 1"); set id "custom" + the code
in lower case, card id "<set id>-<nnn>", numbered in list order. The set.json keeps, per card, the TCGdex card it
was made from with its edits and options (`custom`), so this editor opens the export again (import_set_data).
"""
import datetime
import io
import json
import re
import warnings
import zipfile

from PIL import Image

from . import edit, tcg

FORMAT = "ptcg-sim.custom-set-data"
VERSION = 1
SET_CODE = re.compile(r"[A-Z0-9]{2,5}")
IMAGE_ROOT = "/src/assets/custom-sets"
LOW_SIZE = (245, 342)
STAGE_NAMES = {0: "Basic", 1: "Stage1", 2: "Stage2"}
RARITY = {"common": "Common", "uncommon": "Uncommon", "rare": "Rare"}
MAX_CARDS = 999


def set_id_for(code):
    if not isinstance(code, str) or not SET_CODE.fullmatch(code):
        raise ValueError("set code: 2 to 5 capital letters or digits, like HBW")
    return "custom" + code.lower()


def is_ex(card, layout):
    """An ex card: the Pokemon V -> ex restyle, or a Scarlet & Violet / Mega Evolution ex edited in place."""
    return layout == "ex" or (layout == "modern" and card.get("family") == "ex")


def card_name(card, layout="me"):
    name = f"{card.get('prefix', '')} {card['name']}".strip()
    return f"{name} ex" if is_ex(card, layout) else name


def straight(text):
    """TCGdex writes a straight apostrophe (the card prints a curly one, which edit.printed makes); the simulator's
    parsers read TCGdex text, so the export writes TCGdex's."""
    return (text or "").replace("’", "'")


def _attack(a):
    damage = a.get("damage") or ""
    return {"cost": list(a.get("cost") or []), "name": straight(a.get("name", "")), "effect": straight(a.get("effect")),
            "damage": int(damage) if isinstance(damage, str) and damage.isdigit() else damage}


def _ability(a):
    return {"type": "Ability", "name": straight(a.get("name", "")), "effect": straight(a.get("effect", ""))}


def _side(v):
    return [{"type": v["type"], "value": v.get("value") or "×2"}] if v else []


def mega_stage(card):
    """A Mega's TCGdex stage is the slot it evolves into (me01-003 Mega Venusaur ex: Stage2, evolveFrom Ivysaur):
    one above the pre-evolution's own stage; Stage1 when that cannot be found."""
    pre = (card.get("evolves_from") or "").strip()
    try:
        hits = tcg.search("name", pre) if pre else []
    except Exception as e:                       # noqa: BLE001 - a lookup failure is a note, not a lost export
        warnings.warn(f"{card_name(card)}: could not look up {pre!r} on TCGdex ({e}); stage set to Stage1")
        return "Stage1"
    stages = set()
    for hit in hits[:5]:
        try:
            stages.add(tcg.card(hit["id"]).get("stage"))
        except Exception:                        # noqa: BLE001
            continue
    if "Stage1" in stages or "Stage2" in stages:
        return "Stage2"
    if "Basic" in stages:
        return "Stage1"
    warnings.warn(f"{card_name(card)}: no TCGdex Pokémon named {pre!r}; stage set to Stage1")
    return "Stage1"


def tcgdex_record(card, layout, set_id, local_id, source):
    """One card of the editor (edit.apply's result) as a TCGdex card record for the simulator.
    source: {id, edits, style, attacks, foil} - what made it, kept under "custom" for re-import."""
    rec = {"id": f"{set_id}-{local_id}", "localId": local_id, "name": straight(card_name(card, layout)),
           "image": f"{IMAGE_ROOT}/{set_id}/{local_id}", "illustrator": card.get("illustrator") or "",
           "regulationMark": card.get("reg_mark") or "", "legal": {"standard": False, "expanded": False},
           "custom": {"source": source["id"], "layout": layout, "edits": source.get("edits") or {},
                      "style": source.get("style", "hybrid"), "attacks": bool(source.get("attacks", True)),
                      "foil": bool(source.get("foil"))}}
    if edit.is_trainer(card):
        rec.update(category="Trainer", trainerType=card["kind"],
                   effect="\n\n".join(straight(p) for p in card.get("effect", "").split("\n") if p.strip()),
                   rarity=RARITY.get(card.get("rarity"), "Uncommon"))
        if card.get("hp"):
            rec["hp"] = int(card["hp"])
        if card.get("attacks"):
            rec["attacks"] = [_attack(a) for a in card["attacks"]]
        if card.get("abilities"):
            rec["abilities"] = [_ability(a) for a in card["abilities"]]
        return rec
    stage = card.get("stage", 0)
    rec.update(category="Pokemon", hp=int(card["hp"]), types=[card["type"]],
               stage=STAGE_NAMES.get(stage) or mega_stage(card),
               attacks=[_attack(a) for a in card.get("attacks") or []],
               abilities=[_ability(a) for a in card.get("abilities") or []],
               weaknesses=_side(card.get("weakness")), resistances=_side(card.get("resistance")),
               retreat=int(card.get("retreat") or 0),
               rarity="Double rare" if is_ex(card, layout) else RARITY.get(card.get("rarity"), "Rare"))
    if stage:
        rec["evolveFrom"] = straight(card.get("evolves_from"))
    if is_ex(card, layout):
        rec["suffix"] = "ex"
    if card.get("flavor"):
        rec["description"] = straight(card["flavor"])
    return rec


def set_record(set_id, code, title, cards, today=None):
    date = (today or datetime.date.today()).isoformat()
    return {"format": FORMAT, "version": VERSION, "id": set_id, "code": code, "name": title or code,
            "releaseDate": date, "cards": cards}


def _webp(png_bytes, size=None):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    if size:
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=90, method=4)
    return buf.getvalue()


def export_zip(items, title, code, *, read, render, apply, changelog, today=None):
    """-> zip bytes. items: [{id, edits?, style?, attacks?, foil?}] in set order (the first is card 001).
    read / render / apply / changelog: gui.read, gui.render, edit.apply, gui.changelog (passed in: no import cycle)."""
    set_id = set_id_for(code)
    if not isinstance(items, list) or not items:
        raise ValueError("items: a non-empty list")
    if len(items) > MAX_CARDS:
        raise ValueError(f"items: at most {MAX_CARDS} cards in a set")
    records, images = [], {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("items: expected objects with an id")
            local_id = f"{i + 1:03d}"
            layout, original = read(item["id"])
            card = apply(original, item["edits"]) if item.get("edits") else original
            records.append(tcgdex_record(card, layout, set_id, local_id, item))
            png = render(item["id"], item.get("edits"), item.get("style", "hybrid"), item.get("attacks", True),
                         bool(item.get("foil")))[0]
            images[local_id] = (_webp(png), _webp(png, LOW_SIZE))
    notes = sorted({str(w.message) for w in caught if "torch.jit" not in str(w.message)})
    record = set_record(set_id, code, title, records, today)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{set_id}/set.json", json.dumps(record, indent=1, ensure_ascii=False))
        for local_id, (high, low) in images.items():
            z.writestr(f"{set_id}/{local_id}/high.webp", high, zipfile.ZIP_STORED)
            z.writestr(f"{set_id}/{local_id}/low.webp", low, zipfile.ZIP_STORED)
        z.writestr(f"{set_id}/CHANGELOG.md", changelog(items, title) + (
            "\n\n## Export notes\n" + "".join(f"- {n}\n" for n in notes) if notes else ""))
    return buf.getvalue()


def is_set_data(data):
    return isinstance(data, dict) and data.get("format") == FORMAT and isinstance(data.get("cards"), list)


def import_set_data(data, entry):
    """A set.json this module wrote -> the editor's list with each card's edits and options restored.
    entry(card_id, name, supertype) -> the list entry (gui._set_entry). -> {name, code, cards, edits, options}."""
    if data.get("version") != VERSION:
        raise ValueError(f"custom set version {data.get('version')!r} is not one this editor reads")
    cards, edits, options, seen = [], {}, {}, set()
    for c in data["cards"]:
        custom = c.get("custom") if isinstance(c, dict) else None
        source = custom.get("source") if isinstance(custom, dict) else None
        if not edit.valid_card_id(source) or source in seen:
            continue
        seen.add(source)
        supertype = "Trainer" if c.get("category") == "Trainer" else "Pokémon" if c.get("category") == "Pokemon" else c.get("category")
        cards.append(entry(source, c.get("name", ""), supertype))
        if isinstance(custom.get("edits"), dict) and custom["edits"]:
            edits[source] = custom["edits"]
        options[source] = {"style": custom.get("style", "hybrid"), "attacks": bool(custom.get("attacks", True))}
    code = data.get("code") if isinstance(data.get("code"), str) and SET_CODE.fullmatch(data.get("code", "")) else ""
    return {"name": data.get("name", ""), "code": code, "cards": cards, "edits": edits, "options": options,
            "foil": any(bool((c.get("custom") or {}).get("foil")) for c in data["cards"] if isinstance(c, dict))}
