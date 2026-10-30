"""Custom sets for ptcg-sim: every card of the editor's list as one set the simulator plays.

The zip holds one folder, ready to drop into the simulator's client/src/assets/custom-sets/:

    <set id>/set.json            the set and its cards, in TCGdex's record shape (the simulator's proxy serves
                                 them as TCGdex records, so its deck builder, enrichment and rules engine read
                                 them like any published card)
    <set id>/<nnn>/high.webp     the rendered card, 733x1024
    <set id>/<nnn>/low.webp      the same at 245x342 (the simulator's grid tiles)
    <set id>/CHANGELOG.md        every card's changes from its printed card
    <set id>/foil.json           the TCG Live holofoil of the cards that wear it (the simulator's foil.json, one row
    <set id>/foil/<nnn>-<kind>.webp   per card) and its textures: made with each render, so the foil shows the edits

Set code: 2-5 capital letters or digits, as a decklist prints it ("4 Tyranitar HBW 1"); set id "custom" + the code
in lower case, card id "<set id>-<nnn>", numbered in list order. The set.json keeps, per card, the TCGdex card it
was made from with its edits and options (`custom`), so this editor opens the export again (import_set_data).

write_set_dir puts the same files straight into the simulator's custom-sets folder instead of a zip ("Send to
ptcg-sim"): the simulator watches that folder and reloads the set while it runs. A card whose source and options
match the copy already there keeps its record, images and holofoil, so a push after one edit renders that one card.
A card flagged for holofoil is kept only when its foil is the one its record names (custom.files.foil), so a foil
made outside a push (import-foil.py) is made again by the first push that meets it. It reports its progress and can
be cancelled until it starts writing (a cancel leaves the folder as it was).
existing_cards and dropped_cards tell what a push would take out of the set already there.

The simulator checks a set against its custom-set schema (shared/custom-sets/custom-set-schema.mjs), which lists the
Energy types it takes (ENERGY_TYPES): a card with a type it does not list (Fairy, before the simulator lists it) makes
it reject the whole set, and a set published so vanishes from the site. So write_set_dir refuses such cards, before
rendering, when the simulator's checkout says which types its schema takes (unsupported_types).
"""
import datetime
import hashlib
import io
import json
import os
import re
import shutil
import time
import warnings
import zipfile
from pathlib import Path

from PIL import Image

from . import edit, tcg

FORMAT = "ptcg-sim.custom-set-data"
VERSION = 1
SET_CODE = re.compile(r"[A-Z0-9]{2,5}")
IMAGE_ROOT = "/src/assets/custom-sets"
LOW_SIZE = (245, 342)
FOIL_KINDS = ("front", "etch", "wp", "cf")
FOIL_FILE = re.compile(r"[0-9]{3}-[a-z]+\.webp")
NO_FOIL = "none"                  # custom.files.foil of a card whose render had no holofoil to make: kept as it is
STAGE_NAMES = {0: "Basic", 1: "Stage1", 2: "Stage2"}
RARITY = {"common": "Common", "uncommon": "Uncommon", "rare": "Rare"}
MAX_CARDS = 999
SCHEMA = ("shared", "custom-sets", "custom-set-schema.mjs")      # the simulator's custom-set schema, in its checkout
SCHEMA_TYPES = re.compile(r"ENERGY_TYPES\s*=\s*Object\.freeze\(\[(.*?)\]\)", re.S)


class Cancelled(Exception):
    """The caller asked a build to stop (its `cancelled` callback said so): nothing was written."""


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
    if edit.is_energy(card):
        rec.update(category="Energy", energyType="Special" if card["energy"] == "special" else "Normal",
                   rarity=card.get("rarity_name") or RARITY.get(card.get("rarity"), "Common"))
        if card["energy"] == "special":
            rec["effect"] = "\n\n".join(straight(p) for p in card.get("effect", "").split("\n") if p.strip())
        return rec
    if edit.is_trainer(card):
        rec.update(category="Trainer", trainerType=card["kind"],
                   effect="\n\n".join(straight(p) for p in card.get("effect", "").split("\n") if p.strip()),
                   rarity="ACE SPEC Rare" if card.get("ace") else RARITY.get(card.get("rarity"), "Uncommon"))
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


TRAINER_ORDER = ("Item", "Supporter", "Stadium", "Tool")
POKEMON_STAGES = ("Basic", "Stage1", "Stage2")
# Evolution lines listed under another type than their lowest-stage card's: line (its first card, lowercase) -> type
LINE_TYPES = {"rookidee": "Metal"}
# Cards listed in an evolution line their record does not name (a V restyled as ex is a Basic with no evolveFrom):
# name (lowercase) -> (the line's first card, lowercase; the stage it sorts at, 0-2)
LINE_MEMBERS = {"gallade ex": ("ralts", 2)}


def is_ace_record(rec):
    """Whether a record is an ACE SPEC card: its TCGdex rarity says so."""
    return "ace spec" in str(rec.get("rarity") or "").lower()


def display_order(records, is_ace=None):
    """The records in the order the simulator lists a set: Pokemon (by type A-Z, then evolution line A-Z by its first card, each line
    Basic, Stage 1, Stage 2), then Trainers (Item, Supporter, Stadium, Tool, each A-Z; the ACE SPEC Trainers after them, A-Z), then
    Energy in card-number order (an ACE SPEC Energy stays among it). Only the order of the list: ids and numbers stay. is_ace(source card id) -> bool
    says whether a Trainer whose record does not call itself ACE SPEC is one (records made before it did)."""
    def ace(rec):
        source = _source_of(rec)
        return is_ace_record(rec) or bool(is_ace and source and is_ace(source))

    def stage_of(rec):
        member = LINE_MEMBERS.get(str(rec.get("name") or "").casefold())
        if member:
            return member[1]
        stage = rec.get("stage")
        return POKEMON_STAGES.index(stage) if stage in POKEMON_STAGES else len(POKEMON_STAGES)

    # An evolution line sits together: a Pokemon's line is named by the first card it evolves from, and the line
    # takes the type of its lowest-stage card.
    pokemon = [r for r in records if r.get("category") == "Pokemon"]
    parent = {}
    for r in sorted(pokemon, key=stage_of, reverse=True):
        if r.get("evolveFrom"):
            parent[str(r.get("name") or "").casefold()] = str(r["evolveFrom"]).casefold()

    def line_of(rec):
        name, seen = str(rec.get("name") or "").casefold(), set()
        if name in LINE_MEMBERS:
            return LINE_MEMBERS[name][0]
        while name in parent and name not in seen:
            seen.add(name)
            name = parent[name]
        return name

    lowest = {}
    for r in pokemon:
        line = line_of(r)
        if line not in lowest or (stage_of(r), str(r.get("name") or "")) < (stage_of(lowest[line]), str(lowest[line].get("name") or "")):
            lowest[line] = r

    def key(rec):
        number = number_of(rec)
        name = str(rec.get("name") or "").casefold()
        category = rec.get("category")
        if category == "Pokemon":
            line = line_of(rec)
            if line in LINE_TYPES:
                types = [LINE_TYPES[line]]
            elif stage_of(rec) == 0 and name not in parent.values():      # a Basic nothing evolves from: its own type
                types = rec.get("types") or []
            else:
                types = lowest[line].get("types") or []
            return (0, (0, str(types[0]).casefold()) if types else (1, ""), line, stage_of(rec), name, number)
        if category == "Trainer":
            if ace(rec):
                return (1, 1, 0, name, number)
            kind = rec.get("trainerType")
            return (1, 0, TRAINER_ORDER.index(kind) if kind in TRAINER_ORDER else len(TRAINER_ORDER), name, number)
        return (2, 0, 0, "", number) if category == "Energy" else (3, 0, 0, "", number)
    return sorted(records, key=key)


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


def card_options(item):
    """What a card's render and record depend on besides its source card: compared to reuse an earlier export."""
    return {"edits": item.get("edits") or {}, "style": item.get("style", "hybrid"),
            "attacks": bool(item.get("attacks", True)), "foil": bool(item.get("foil"))}


def build_set(items, title, code, *, read, render, apply, changelog, today=None, reuse=None, progress=None,
              cancelled=None, is_ace=None):
    """-> (set id, {path in the set folder: bytes}, {"rendered": n, "reused": n}): set.json, the cards' WebPs and
    CHANGELOG.md. items: [{id, edits?, style?, attacks?, foil?}] in set order (the first is card 001).
    read / render / apply / changelog: gui.read, gui.render, edit.apply, gui.changelog (passed in: no import cycle).
    reuse(item) -> (record, high webp, low webp, foil) | None: a card exported before with the same options, kept
    instead of rendered and numbered for its place in this list (foil: foil_from_holo's shape, None without holofoil).
    A card flagged for holofoil also gets its foil textures under foil/ and a row in foil.json.
    progress("rendering", cards done, cards in all, id of the card being made) is called before each card and once
    at the end; cancelled() -> True before a card stops the build (Cancelled).
    set.json lists the cards in display order (display_order; is_ace: its callback), numbered by `items`' order."""
    set_id = set_id_for(code)
    if not isinstance(items, list) or not items:
        raise ValueError("items: a non-empty list")
    if len(items) > MAX_CARDS:
        raise ValueError(f"items: at most {MAX_CARDS} cards in a set")
    records, files, counts, foil_rows = [], {}, {"rendered": 0, "reused": 0}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, item in enumerate(items):
            if cancelled and cancelled():
                raise Cancelled
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("items: expected objects with an id")
            if progress:
                progress("rendering", i, len(items), item["id"])
            local_id = f"{i + 1:03d}"
            kept = reuse(item) if reuse else None
            if kept:
                record, high, low, foil = kept
                records.append({**record, "id": f"{set_id}-{local_id}", "localId": local_id,
                                "image": f"{IMAGE_ROOT}/{set_id}/{local_id}"})
                counts["reused"] += 1
            else:
                layout, original = read(item["id"])
                card = apply(original, item["edits"]) if item.get("edits") else original
                records.append(tcgdex_record(card, layout, set_id, local_id, item))
                made = render(item["id"], item.get("edits"), item.get("style", "hybrid"), item.get("attacks", True),
                              bool(item.get("foil")))
                high, low = _webp(made[0]), _webp(made[0], LOW_SIZE)
                digests, foil = {"high": _digest(high), "low": _digest(low)}, None
                if item.get("foil"):
                    foil = foil_from_holo(made[3] if len(made) > 3 else None)
                    digests["foil"] = foil_digest(foil) if foil else NO_FOIL
                    if not foil:
                        warnings.warn(f"{records[-1]['name']}: no holofoil could be made; the card is set without it")
                records[-1]["custom"]["files"] = digests
                counts["rendered"] += 1
            files[f"{local_id}/high.webp"], files[f"{local_id}/low.webp"] = high, low
            if foil:
                foil_rows[local_id] = foil_row(foil, local_id)
                for kind, data in foil["textures"].items():
                    files[f"foil/{local_id}-{kind}.webp"] = data
    if progress:
        progress("rendering", len(items), len(items), "")
    notes = sorted({str(w.message) for w in caught if "torch.jit" not in str(w.message)})
    record = set_record(set_id, code, title, display_order(records, is_ace), today)
    files["CHANGELOG.md"] = (changelog(items, title) + (
        "\n\n## Export notes\n" + "".join(f"- {n}\n" for n in notes) if notes else "")).encode()
    if foil_rows:
        files["foil.json"] = (json.dumps({"v": 1, "cards": foil_rows}, indent=1) + "\n").encode()
    files["set.json"] = json.dumps(record, indent=1, ensure_ascii=False).encode()
    return set_id, files, counts


def export_zip(items, title, code, *, read, render, apply, changelog, today=None, is_ace=None):
    """-> zip bytes: build_set's files under one <set id>/ folder."""
    set_id, files, _ = build_set(items, title, code, read=read, render=render, apply=apply, changelog=changelog,
                                 today=today, is_ace=is_ace)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(f"{set_id}/{name}", data, zipfile.ZIP_STORED if name.endswith(".webp") else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


CARD_DIR = re.compile(r"[0-9]{3}")


def _digest(data):
    return hashlib.sha1(data).hexdigest()[:16]


def _foil_webp(png, kind):
    img = Image.open(io.BytesIO(png))
    img.load()
    buf = io.BytesIO()
    if kind == "front":
        img.convert("RGB").save(buf, "WEBP", quality=80, method=4)       # lossy is invisible at 1024 px
    else:
        img.save(buf, "WEBP", quality=92, method=4)                      # the plates are soft shader inputs
    return buf.getvalue()


def foil_from_holo(holo):
    """What a set keeps of a render's holofoil. holo: gui.Render's fourth part, (foil.json record, {file name: PNG
    bytes}, ...) or None. -> {tcgl, shader, material, cc, textures: {kind: WebP bytes}}: the base printing when TCG
    Live foils the card, else its reverse holo's (the choice import-foil.py makes). None when there is no foil, or
    the record names a texture the render did not write."""
    if not holo:
        return None
    record, written = holo[0], holo[1]
    chosen = next((record[key] for key in ("base", "reverse") if isinstance(record.get(key), dict)), None)
    if not chosen or not chosen.get("shader") or not chosen.get("material"):
        return None
    names = {"front": record.get("front"), **{kind: chosen.get(kind) for kind in FOIL_KINDS if kind != "front"}}
    textures = {}
    for kind in FOIL_KINDS:
        if not names.get(kind):
            continue
        if names[kind] not in written:
            return None
        textures[kind] = _foil_webp(written[names[kind]], kind)
    if "front" not in textures:
        return None
    return {"tcgl": record.get("tcglive") or "", "shader": chosen["shader"], "material": chosen["material"],
            "cc": chosen.get("cc") or None, "textures": textures}


def foil_digest(foil):
    """Names a card's foil in custom.files.foil: its shader, material, pattern and every texture's bytes."""
    return _digest(json.dumps([foil["tcgl"], foil["shader"], foil["material"], foil["cc"],
                               sorted((kind, _digest(data)) for kind, data in foil["textures"].items())]).encode())


def foil_row(foil, local_id):
    """The card's row in foil.json: the simulator's shape, its textures named <nnn>-<kind>.webp."""
    row = {"tcgl": foil["tcgl"], "shader": foil["shader"], "material": foil["material"]}
    for kind in FOIL_KINDS:
        if kind in foil["textures"]:
            row[kind] = f"{local_id}-{kind}.webp"
    if foil["cc"]:
        row["cc"] = foil["cc"]
    return row


def _read_foil(set_dir, local_id, rows):
    """The foil kept for card `local_id` of set_dir (foil_from_holo's shape), or None when its row is missing or
    incomplete, or a texture it names is gone."""
    row = rows.get(local_id)
    if not isinstance(row, dict) or not row.get("shader") or not row.get("material"):
        return None
    textures = {}
    for kind in FOIL_KINDS:
        if not row.get(kind):
            continue
        try:
            textures[kind] = (Path(set_dir) / "foil" / f"{local_id}-{kind}.webp").read_bytes()
        except OSError:
            return None
    if "front" not in textures:
        return None
    return {"tcgl": row.get("tcgl") or "", "shader": row["shader"], "material": row["material"],
            "cc": row.get("cc") or None, "textures": textures}


def _foil_rows(set_dir):
    try:
        data = json.loads((Path(set_dir) / "foil.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = data.get("cards") if isinstance(data, dict) else None
    return rows if isinstance(rows, dict) else {}


def exported_cards(set_dir, left_out=None):
    """{source card id: (record, high webp, low webp, foil)} for the cards of a set this module wrote into set_dir
    that a push can keep instead of rendering; {} when there is no such set or it cannot be read (the push then
    renders every card). foil is the card's holofoil as foil_from_holo shapes it, None for a card without.
    A card whose images are not the ones its record names (custom.files: a push that stopped half way, a hand edit)
    is left out, so it is rendered again. So is a card flagged for holofoil whose foil is not the one its record names
    (custom.files.foil): a foil made outside a push (import-foil.py) is not known to match the card's edits. A record
    with no custom.files (a set exported before records kept them) has its images taken as they are, and comes back
    with their digests, so the next push checks them. left_out, a dict, is filled with {source id: "images" | "foil"}
    for the cards left out."""
    try:
        data = json.loads((Path(set_dir) / "set.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not is_set_data(data):
        return {}
    rows = _foil_rows(set_dir)
    out = {}
    for rec in data["cards"]:
        custom = rec.get("custom") if isinstance(rec, dict) else None
        local_id = rec.get("localId") if isinstance(rec, dict) else None
        if not isinstance(custom, dict) or not isinstance(local_id, str) or not CARD_DIR.fullmatch(local_id):
            continue
        source = custom.get("source")
        if not isinstance(source, str):
            continue
        try:
            high = (Path(set_dir) / local_id / "high.webp").read_bytes()
            low = (Path(set_dir) / local_id / "low.webp").read_bytes()
        except OSError:
            continue
        digests = {"high": _digest(high), "low": _digest(low)}
        files = custom.get("files")
        if not isinstance(files, dict):
            files = digests
            rec = {**rec, "custom": {**custom, "files": digests}}
        elif {k: files.get(k) for k in digests} != digests:
            if left_out is not None:
                left_out.setdefault(source, "images")
            continue
        foil = None
        if custom.get("foil") and files.get("foil") != NO_FOIL:
            foil = _read_foil(set_dir, local_id, rows)
            if not foil or files.get("foil") != foil_digest(foil):
                if left_out is not None:
                    left_out.setdefault(source, "foil")
                continue
        out.setdefault(source, (rec, high, low, foil))
    for source in out:                                  # a source listed twice: its kept copy counts
        if left_out is not None:
            left_out.pop(source, None)
    return out


def _write_if_changed(path, data, attempts=5):
    """-> whether the file was written. Written whole to a temporary name, then renamed over the old one, so a
    reader never sees half a file; Windows refuses the rename while another program holds the file, so it retries."""
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
    except OSError:
        tmp.unlink(missing_ok=True)              # disk full: no half file left behind
        raise
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.1 * (attempt + 1))
    return True


def number_of(rec):
    """A record's card number (its localId), or a number past every card's when it has none."""
    local = rec.get("localId") if isinstance(rec, dict) else None
    return int(local) if isinstance(local, str) and local.isdigit() else MAX_CARDS + 1


def by_number(cards):
    """The records in card-number order (a stable sort: records without a number keep their places at the end). The
    set.json lists them in display order (display_order); numbers, and the position logic of a push, follow the
    numbers."""
    return sorted(cards, key=number_of)


def _set_json(set_dir):
    """The set.json in set_dir as a dict with a "cards" list in card-number order, or None when there is none that can
    be read."""
    try:
        data = json.loads((Path(set_dir) / "set.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(data, dict) and isinstance(data.get("cards"), list)):
        return None
    return {**data, "cards": by_number(data["cards"])}


def _source_of(rec):
    custom = rec.get("custom") if isinstance(rec, dict) and isinstance(rec.get("custom"), dict) else {}
    return custom.get("source") if isinstance(custom.get("source"), str) else None


def existing_cards(set_dir):
    """[(source card id or None, card name)] of the set.json already in set_dir, in its order; None when there is
    none that can be read. Any custom set counts, not only one this module wrote (its cards then name no source)."""
    data = _set_json(set_dir)
    if data is None:
        return None
    return [(_source_of(rec), str(rec.get("name") or rec.get("id") or "?") if isinstance(rec, dict) else "?")
            for rec in data["cards"]]


def unlisted_items(set_dir, ids):
    """[(place, item)] of the cards in the set already in set_dir that `ids` (source card ids) do not name, in the
    set's order, each as an item a push can send again: the card it was made from with the edits and options its
    record keeps (custom). A card that names no source (a set this module did not write) cannot be sent again and is
    not listed. place: where the card is in the set now (0-based)."""
    data = _set_json(set_dir)
    held, listed = [], set(ids)
    for place, rec in enumerate(data["cards"] if data else []):
        source = _source_of(rec)
        if not source or source in listed:
            continue
        custom = rec["custom"]
        held.append((place, {"id": source, "edits": custom.get("edits") or {}, "style": custom.get("style") or "hybrid",
                             "attacks": custom.get("attacks") is not False, "foil": bool(custom.get("foil"))}))
        listed.add(source)
    return held


def dropped_cards(existing, ids):
    """[(source, name)] of the cards in `existing` (existing_cards) that a push of the list `ids` (the cards' source
    ids) would not carry over. Cards are matched by the card they were made from; when some name none (a set this
    module did not write) the two are compared by position alone: whatever lies past the end of the list goes."""
    keep = set(ids)
    if all(source for source, _ in existing):
        return [(source, name) for source, name in existing if source not in keep]
    return existing[len(ids):]


def plan_set(items, set_dir):
    """What a push of `items` (in set order) would change in the set already in set_dir: nothing is rendered or
    written. -> {existing: cards there (None: no set), added: [item], changed: [(item, the options its card was pushed
    with, whether its images are not the ones its record names)], removed: [(source, name)], moved: cards that get
    another number, unchanged: n}. A card is unchanged when the push keeps it (exported_cards: the same test, so the
    plan is what the push does) and its options (card_options) are the ones its record keeps; any other card the push
    renders is changed. The third part of a changed card is False when its options changed, True when its images in
    the set are not the ones its record names, "foil" when its holofoil is not known to match it."""
    data = _set_json(set_dir)
    left_out = {}
    kept = exported_cards(set_dir, left_out)
    records = {}
    for place, rec in enumerate(data["cards"] if data else []):
        if _source_of(rec):
            records.setdefault(_source_of(rec), (place, rec))
    plan = {"existing": len(data["cards"]) if data else None, "added": [], "changed": [], "removed": [], "moved": 0,
            "unchanged": 0}
    for place, item in enumerate(items):
        if item["id"] not in records:
            plan["added"].append(item)
            continue
        was, rec = records[item["id"]]
        plan["moved"] += was != place
        custom = rec["custom"]
        old = {k: custom.get(k) for k in ("edits", "style", "attacks", "foil")}
        if item["id"] not in kept:
            plan["changed"].append((item, old, "foil" if left_out.get(item["id"]) == "foil" else True))
        elif old == card_options(item):
            plan["unchanged"] += 1
        else:
            plan["changed"].append((item, old, False))
    if data:
        plan["removed"] = dropped_cards(existing_cards(set_dir), [item["id"] for item in items])
    return plan


def existing_title(set_dir):
    """The name of the set already in set_dir ("My Custom Set"), or "" when there is none."""
    data = _set_json(set_dir)
    return data["name"] if data and isinstance(data.get("name"), str) else ""


def schema_types(root):
    """The Energy types the simulator's custom-set schema takes, read from its checkout (root: its custom-sets folder,
    client/src/assets/custom-sets); None when the checkout's schema is not there."""
    path = Path(root).resolve().parents[3].joinpath(*SCHEMA) if len(Path(root).resolve().parents) > 3 else None
    if path is None or not path.is_file():
        return None
    m = SCHEMA_TYPES.search(path.read_text(encoding="utf-8"))
    return set(re.findall(r"['\"](\w+)['\"]", m.group(1))) if m else None


def card_types(card):
    """Every Energy type a card's record names that the simulator's schema checks: a Pokemon's type, costs, weakness
    and resistance; a Trainer's attack costs (Technical Machines). An Energy card's record names none (its type and
    what it provides stay in its text), and a Basic Energy is not exported."""
    if edit.is_energy(card) or edit.is_trainer(card):
        return {t for a in card.get("attacks") or [] for t in a.get("cost") or []}
    return ({card["type"]} | {t for a in card.get("attacks") or [] for t in a.get("cost") or []}
            | {v["type"] for v in (card.get("weakness"), card.get("resistance")) if v})


def unsupported_types(items, root, *, read, apply):
    """Cards of items naming an Energy type the simulator's schema does not take: [(card name, [types])], or []
    when the schema takes them all or cannot be read (an older checkout's set is checked by the simulator itself)."""
    taken = schema_types(root)
    if taken is None:
        return []
    out = []
    for item in items:
        layout, card = read(item["id"])
        card = apply(card, item.get("edits")) if item.get("edits") else card
        missing = sorted(card_types(card) - taken)
        if missing:
            out.append((card_name(card, layout), missing))
    return out


def write_set_dir(items, title, code, root, *, read, render, apply, changelog, force=False, today=None, progress=None,
                  cancelled=None, is_ace=None):
    """The list as a custom set written straight into the simulator's custom-sets folder (root): <root>/<set id>/.
    A card whose source and options (card_options) match the set already there keeps its record, images and
    holofoil, unless force. Only files whose bytes changed are written, set.json last (the simulator reloads on it);
    card folders past the end of the list are removed, and so are foil.json and the foil/ textures no card of the
    list uses any more (the foil rows follow the cards: a row left for a number would put its foil on another card).
    Anything else in the folder is left.
    progress / cancelled: as build_set's, and progress("writing", files done, files in all, path written) while the
    files go out. A cancel counts until the first file is written (Cancelled: the folder is as it was); then the
    push finishes.
    A card naming an Energy type the simulator's schema does not take (unsupported_types) is refused (ValueError)
    before anything is rendered: the simulator would reject the whole set.
    -> {setId, rendered, reused, written: [path], unchanged: n, removed: [card folder]}"""
    refused = unsupported_types(items, root, read=read, apply=apply)
    if refused:
        types = ", ".join(sorted({t for _, missing in refused for t in missing}))
        names = "; ".join(f"{name} ({', '.join(missing)})" for name, missing in refused[:5])
        more = " …" if len(refused) > 5 else ""
        raise ValueError(f"the simulator's custom-set schema does not take {types} yet and would reject the whole set: "
                         f"{names}{more}. Add it to ENERGY_TYPES in the simulator's {'/'.join(SCHEMA)} (and deploy "
                         "that before publishing), or change these cards' types")
    set_dir = Path(root) / set_id_for(code)
    before = {} if force else exported_cards(set_dir)

    def reuse(item):
        kept = before.get(item.get("id"))
        if not kept:
            return None
        custom = kept[0]["custom"]
        return kept if {k: custom.get(k) for k in ("edits", "style", "attacks", "foil")} == card_options(item) else None

    set_id, files, counts = build_set(items, title, code, read=read, render=render, apply=apply,
                                      changelog=changelog, today=today, reuse=reuse, progress=progress,
                                      cancelled=cancelled, is_ace=is_ace)
    if cancelled and cancelled():
        raise Cancelled
    written, unchanged = [], 0
    if progress:
        progress("writing", 0, len(files), "")
    for done, name in enumerate(sorted(files, key=lambda n: n == "set.json"), 1):
        if _write_if_changed(set_dir / name, files[name]):
            written.append(name)
        else:
            unchanged += 1
        if progress:
            progress("writing", done, len(files), name)
    keep = {name.split("/")[0] for name in files if "/" in name}
    removed = sorted(d.name for d in set_dir.iterdir() if d.is_dir() and CARD_DIR.fullmatch(d.name) and d.name not in keep)
    for name in removed:
        shutil.rmtree(set_dir / name)
    _prune_foil(set_dir, files)
    return {"setId": set_id, **counts, "written": written, "unchanged": unchanged, "removed": removed}


def _prune_foil(set_dir, files):
    """Removes what a push's foil left behind: foil/ textures no row names, and foil.json when no card has foil."""
    if "foil.json" not in files:
        (set_dir / "foil.json").unlink(missing_ok=True)
    foil_dir = set_dir / "foil"
    if not foil_dir.is_dir():
        return
    named = {name[len("foil/"):] for name in files if name.startswith("foil/")}
    for path in foil_dir.iterdir():
        if path.is_file() and FOIL_FILE.fullmatch(path.name) and path.name not in named:
            path.unlink()
    if not any(foil_dir.iterdir()):
        foil_dir.rmdir()


def is_set_data(data):
    return isinstance(data, dict) and data.get("format") == FORMAT and isinstance(data.get("cards"), list)


def import_set_data(data, entry):
    """A set.json this module wrote -> the editor's list with each card's edits and options restored.
    entry(card_id, name, supertype) -> the list entry (gui._set_entry). -> {name, code, cards, edits, options}."""
    if data.get("version") != VERSION:
        raise ValueError(f"custom set version {data.get('version')!r} is not one this editor reads")
    cards, edits, options, seen = [], {}, {}, set()
    for c in by_number(data["cards"]):                    # the list keeps the numbers' order, not the display order
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
