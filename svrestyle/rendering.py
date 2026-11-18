"""A card read and rendered, for every part of the editor: the layout it gets, its render (cached by request, with
its holofoil), what differs from its printed card and the changelog of a list, and the entry a set list gives it.
The HTTP server (gui.py), the projects, sends and downloads all take their cards from here.
"""
import datetime
import hashlib
import io
import json
import threading
import warnings
from collections import OrderedDict, namedtuple

import numpy as np
from PIL import Image

from . import cache, compose, edit, energy, export, foil, layers, mecompose, modern, tcg, trainer

STYLES = ("hybrid", "panel")

_render_lock = threading.Lock()       # LaMa, fonts and the asset builders are not thread-safe
_renders = OrderedDict()              # request key -> Render; the last few, for zip downloads
_renders_lock = threading.Lock()      # the cache's own (not _render_lock): a hit must not wait for a render
_RENDER_CACHE = 48


class UserError(ValueError):
    """A problem with the request, shown to the user as is."""


class Render(namedtuple("Render", "png notes layout holo", defaults=(None,))):
    """(png bytes, warnings, layout, holo): holo is a Holo when the holofoil was asked for and made, else None."""


class Holo(namedtuple("Holo", "record files plates token")):
    """A render's holofoil: record, the foil.json record; files, {file name: bytes} as foil.files writes them; plates,
    {"base" | "reverse": white plate PNG at card size, for the preview}; token, what its texture URLs carry."""


# ---------------------------------------------------------------- cards
def layout_of(card_id):
    if not edit.valid_card_id(card_id):
        raise UserError(f"{card_id!r} is not a TCGdex card id (like swsh6-99 or sm10-19)")
    try:
        data = tcg.card(card_id)
    except (ValueError, cache.DownloadError) as e:
        raise UserError(str(e)) from e
    return layout_for(card_id, data)


def layout_for(card_id, data):
    """The layout a card gets, from its TCGdex record (category and name: a set list's supertype and name do): the one
    place that decides, for the GUI, the set lists and the command line."""
    if data.get("category") == "Trainer":       # a Scarlet & Violet Trainer too: set again on its own kind's frame
        return "trainer"
    if data.get("category") == "Energy":        # read_card refuses Scarlet & Violet and Mega Evolution Energy
        return "energy"
    if modern.is_modern(card_id):
        return "modern"
    return "ex" if str(data.get("name") or "").endswith(" V") else "me"


def read(card_id):
    layout = layout_of(card_id)
    reader = {"ex": compose.read_card, "me": mecompose.read_card, "modern": modern.read_card,
              "trainer": trainer.read_card, "energy": energy.read_card}[layout]
    try:
        card = reader(card_id)
    except SystemExit as e:                     # the composers refuse unsupported cards this way
        raise UserError(str(e)) from e
    except cache.DownloadError as e:
        raise UserError(str(e)) from e
    return layout, card


def _png(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode).save(buf, "PNG")
    return buf.getvalue()


def render(card_id, edits=None, style="hybrid", attacks=True, holo=False):
    """-> Render; cached by request. holo: also re-make the card's TCG Live holofoil for this render: the plates
    follow every edit (HP, costs, Ability badge, weakness, resistance and retreat icons are cut out where drawn)."""
    if style not in STYLES:
        raise UserError(f"style: one of {', '.join(STYLES)}")
    if holo and not tcg.TCGLIVE:
        raise UserError("holofoil needs a Pokemon TCG Live extract: start the GUI with --tcglive DIR "
                        "(or set SVRESTYLE_TCGLIVE)")
    key = json.dumps([card_id, edits, style, bool(attacks), bool(holo)], sort_keys=True, ensure_ascii=False)
    with _renders_lock:
        if key in _renders:
            _renders.move_to_end(key)
            return _renders[key]
    with _render_lock, warnings.catch_warnings(record=True) as caught, layers.recording() as drawn:
        warnings.simplefilter("always")
        layout, original = read(card_id)        # its warnings shown too (a Trainer's kind guessed, its text missing)
        card = edit.apply(original, edits) if edits else original
        if layout == "energy":
            img, _ = energy.compose_card(card_id, card=card)
            built = energy.build_foil(card, img, drawn) if holo else None
        elif layout == "trainer":
            img, _ = trainer.compose_card(card_id, card=card)
            built = trainer_foil(card_id, card, img, drawn, original) if holo else None
        elif layout == "modern":
            img, report = modern.compose_card(card_id, card, original)
            built = foil.build_modern(card_id, card, img, drawn, original) if holo else None
        else:
            if layout == "ex":
                img, _ = compose.compose(card_id, compose.Options(style=style, attacks=bool(attacks)), card=card)
            else:
                img, _ = mecompose.compose_card(card_id, bool(attacks), card=card)
            built = foil.build(card_id, layout, img, drawn, card) if holo else None
    notes = export.warning_notes(caught)
    made = None
    if built:
        record, written = foil.files(built, f"{card_id}-{layout}", card_id)
        plates = {k: _png(foil.card_space(built[k][1]), "L") for k in ("base", "reverse") if built[k]}
        made = Holo(record, written, plates, hashlib.sha256(key.encode()).hexdigest()[:20])
    result = Render(_png(img), notes, layout, made)
    with _renders_lock:
        _renders[key] = result
        while len(_renders) > _RENDER_CACHE:
            _renders.popitem(last=False)
    return result


def trainer_foil(card_id, card, img, drawn, original):
    """A Trainer's holofoil, by what its printed card is (original: read_card's, before edits): an ACE SPEC card's
    own printing, edited; a Black & White ACE SPEC set on the Scarlet & Violet frame, the converted one's; any other,
    the Gen 8 reverse holo. The one place that chooses, for the GUI and the command line."""
    if original.get("ace"):
        return foil.build_ace(card_id, card, img, drawn, original)
    if original.get("ace_convert"):
        return foil.build_ace_convert(card, img, drawn)
    return foil.build_trainer(card_id, card, img, drawn)


LAYOUT_NOTES = {"ex": "Restyled as a Scarlet & Violet ex card", "me": "Restyled as a Mega Evolution card",
                "trainer": "Restyled as a Scarlet & Violet Trainer", "energy": "Restyled as a Mega Evolution Energy"}


def card_changes(card_id, edits=None, style="hybrid", attacks=True):
    """-> (printed name, [what differs from the printed card]): the restyle itself, then edit.changes."""
    layout, original = read(card_id)
    card = edit.apply(original, edits) if edits else original
    notes = []
    if layout in LAYOUT_NOTES and not modern.is_modern(card_id):     # a Gen 9 Trainer is set again, not restyled
        notes.append(LAYOUT_NOTES[layout] + (" (translucent type panel)" if layout == "ex" and style == "panel" else ""))
        if not attacks and layout not in ("trainer", "energy"):
            notes.append("Abilities and attacks left off")
    return tcg.card(card_id).get("name", card_id), notes + edit.changes(original, card)


def changelog(items, title=""):
    """Markdown: per card what was changed; cards left as printed listed together at the end."""
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines, unchanged = [f"# {title.strip() if isinstance(title, str) and title.strip() else 'Cards'}: changelog", "",
                        f"Changes from each printed card, {stamp}.", ""], []
    for it in items:
        if not isinstance(it, dict):
            raise UserError("items: expected objects")
        name, changed = card_changes(it.get("id"), it.get("edits"), it.get("style", "hybrid"), it.get("attacks", True))
        if changed:
            lines += [f"## {name} ({it['id']})", "", *[f"- {c}" for c in changed], ""]
        else:
            unchanged.append(f"{name} ({it['id']})")
    if unchanged:
        lines += ["## Unchanged", "", *[f"- {u}" for u in unchanged], ""]
    return "\n".join(lines)


def _is_ace(cid):
    """Whether a Scarlet & Violet Trainer is an ACE SPEC card (its holofoil can be made): True or False, None when
    TCGdex cannot say (not known: a list entry does not keep it, so the next load asks again)."""
    if not modern.is_modern(cid):
        return False
    try:
        return trainer.ACE_RARITY in (tcg.card(cid).get("rarity") or "").lower()
    except (ValueError, cache.DownloadError):
        return None


def _trainer_entry(entry):
    """A Trainer's list entry with gen9 and ace; ace left out while TCGdex cannot say (_is_ace)."""
    ace = _is_ace(entry.get("id"))
    return {**entry, "gen9": modern.is_modern(entry.get("id")), **({} if ace is None else {"ace": ace})}


def _set_entry(cid, name, supertype):
    """A set list's entry: the layout the card gets, or why it is skipped. supertype: the export's or TCGdex's."""
    entry = {"id": cid, "name": name}
    if supertype not in (None, "Pokémon", "Pokemon", "Trainer", "Energy"):
        return {**entry, "skip": "not a Pokémon, Trainer or Energy card"}
    entry["layout"] = layout_for(cid, {"category": supertype, "name": name})   # Energy: SV / ME kept as printed
    return _trainer_entry(entry) if entry["layout"] == "trainer" else entry


def _current_entries(cards):
    """A saved list as this version sorts it: projects keep each card's entry from their import, so a card once
    skipped (Trainers were, before they could be edited) is looked at again. A card TCGdex can't give stays as saved."""
    out = []
    for c in cards:
        if isinstance(c, dict) and c.get("skip") and edit.valid_card_id(c.get("id")):
            try:
                c = _set_entry(c["id"], c.get("name", ""), tcg.card(c["id"]).get("category"))
            except (ValueError, cache.DownloadError):
                pass
        elif isinstance(c, dict) and c.get("layout") == "trainer" and ("gen9" not in c or "ace" not in c):
            c = _trainer_entry(c)
        out.append(c)
    return out
