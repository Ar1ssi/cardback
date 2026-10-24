"""Local web GUI: load a card, edit its name, HP, abilities, attacks (costs, damage, text), weakness, resistance and
retreat, and render it in the ex or Mega Evolution layout (Scarlet & Violet and Mega Evolution cards: edited in
place, see modern.py), with its TCG Live holofoil re-made to fit the edits; or load a Trainer (Item, Supporter,
Stadium, Pokemon Tool), edit its name, type and text, and render it as a Scarlet & Violet Trainer (trainer.py;
a Scarlet & Violet or Mega Evolution Trainer keeps its art and is set again on its kind's frame); or import a custom-set export and restyle or edit every Pokemon and Trainer in it. Standard library
only; serves on 127.0.0.1.

  python -m svrestyle gui [--port 8765] [--fonts DIR] [--tcglive DIR] [--ptcg-sim DIR] [--no-browser]

API (JSON in and out unless noted):
  GET  /                      the page (gui.html)
  GET  /api/meta              energy types, limits, weakness/resistance values, whether holofoil is available,
                              the wording terms the page capitalises as you type, whether /api/suggest works,
                              whether a ptcg-sim folder is set for /api/push-set (ptcgSim)
  POST /api/suggest           {text, kind: "attack"|"ability"|"trainer", card?: {name, type, stage, hp}} ->
                              {text, notes: [str]}: the text rewritten in printed-card wording by Claude
                              (wording.suggest: ANTHROPIC_API_KEY, else the Claude Code CLI's login)
  GET  /api/card?id=ID        {id, layout: "ex"|"me"|"modern"|"trainer", name, card: editable fields}
  POST /api/render            {id, edits?, style?, attacks?, foil?} -> {png: base64, warnings, layout, changes,
                              foil?: {front: URL, base, reverse: {png: base64 white plate (card size), shader,
                              material, cc, textures: {wp, etch?, cf?: URL}} | null}}
                              changes: what differs from the printed card, one line each (edit.changes)
  GET  /api/foil/file/T/NAME  a texture of the render whose foil answer gave token T (1024x1024 PNG)
  GET  /api/foil/shaders.json the game's TPCi/Cards3D shaders as GLSL ES 3.00 (from the extract's WebApp)
  GET  /api/foil/materials.json  its MAT_Cards3D_* materials, texture URLs under /api/foil/tex/
  GET  /api/foil/tex/NAME.png the materials' textures (spectrum, shine, glitter, patterns), lossless
  GET  /api/original/ID.png   the printed card (the scan the render starts from), for comparing
  GET  /api/icon/TYPE.png     the energy icon used on cards (PNG)
  POST /api/set               a ptcg-sim custom-set export ({cards: [{id, name, supertype}]}) -> {name, cards:
                              [{id, name, layout | skip}]}; or a set.json this editor exported (export.py) ->
                              also {code, edits, options, foil}: the cards with their edits restored
  POST /api/export-set        {items: [{id, edits?, style?, attacks?, foil?}], title?, code} -> application/zip:
                              the set for ptcg-sim (<set id>/set.json, <nnn>/high.webp + low.webp, CHANGELOG.md;
                              see export.py)
  POST /api/push-set          {items, title?, code, force?} -> {setId, rendered, reused, written: [path], unchanged,
                              removed: [card folder]}: the same set written into the simulator's custom-sets folder
                              (SVRESTYLE_PTCG_SIM), which the running simulator reloads; only cards changed since
                              the last push are rendered (export.write_set_dir), every card with force
  POST /api/zip              {items: [{id, edits?, style?, attacks?, foil?}], title?} -> application/zip (with foil
                              also the 1024x1024 front, the white plates and <card>-foil.json, see foil.files), and
                              CHANGELOG.md
  POST /api/changelog         {items, title?} -> text/markdown: every card's changes from its printed card
  GET  /api/projects          {projects: [{name, saved, cards}]}, newest first
  GET  /api/projects?name=N   {name, saved, rev, data}: a saved project (the page's card list, edits and options)
  POST /api/projects          {name, data, client?} -> {name, saved, rev}: save (overwrites a project of that name)
  POST /api/projects/patch    {name, patch, client?} -> {name, saved, rev}: change only what the patch names (see
                              apply_patch), making the project if there is none
  POST /api/projects/delete   {name, renamed?, client?} -> {deleted: name}
  GET  /api/projects/poll?name=N&after=SEQ&client=C  long poll for what happens to the project: answers when there
                              is news past SEQ (or after POLL_WAIT s with none) -> {seq, viewers, events: [{event,
                              data}]}: patch {rev, client, patch}, replaced {rev, client}, renamed {to, client},
                              deleted {client}, viewers {viewers}. after=-1 (or a SEQ it no longer has) -> {seq,
                              viewers, events: [], hello: {rev}} at once: the page reads the project again if its
                              rev differs. (Long polls, not an event stream: Cloudflare quick tunnels hold streams.)
  POST /api/projects/leave    {name, client}: the page is closing (it stops counting as a viewer at once)

Projects are JSON files under the cache folder (SVRESTYLE_CACHE/projects), so work resumes after a restart. Each
save or patch raises the project's rev; a page that missed some (reconnecting) reads the project again.
"""
import base64
import datetime
import functools
import hashlib
import io
import json
import os
import re
import threading
import time
import warnings
import webbrowser
import zipfile
from collections import OrderedDict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import cache, compose, edit, export, foil, layers, mecompose, meframe, modern, tcg, trainer, wording

PAGE = Path(__file__).with_name("gui.html")
STATIC = Path(__file__).with_name("static")
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
# The extract's foil data: the WebApp's translated shaders and materials, and the shaders bundle's textures as
# lossless PNGs (the WebApp's own copies, lossy WebP, when those are missing).
FOIL_DATA = ("WebApp", "web", "data", "foil")
FOIL_TEX = ("pokemon_Pokemon TCG Live", "shadersbundle", "Texture2D")
FOIL_TEX_WEB = ("WebApp", "web", "assets", "foil", "tex")
TEXTURE_NAME = re.compile(r"[A-Za-z0-9_]{1,80}")
FILE_NAME = re.compile(r"[\w.-]{1,120}\.png")
MAX_BODY = 8 << 20
MAX_ZIP_ITEMS = 60
STYLES = ("hybrid", "panel")
PROJECT_NAME = re.compile(r"\w[\w .,'()&+#-]{0,59}")
MAX_PROJECT = 4 << 20                 # bytes of one saved project
POLL_WAIT = 20                        # s a poll waits for news before answering with none (tunnels cut at ~100 s)
VIEWER_GONE = POLL_WAIT + 25          # s without a poll after which a page no longer counts as having it open
EVENT_LOG = 200                       # events each project keeps for pages catching up

SIM_SETS = ("client", "src", "assets", "custom-sets")   # where the simulator keeps custom sets, in its checkout

_render_lock = threading.Lock()       # LaMa, fonts and the asset builders are not thread-safe
_push_lock = threading.Lock()         # one write into the simulator's folder at a time
_renders = OrderedDict()              # request key -> Render; the last few, for zip downloads
_RENDER_CACHE = 48


class UserError(ValueError):
    """A problem with the request, shown to the user as is."""


class Render(tuple):
    """(png bytes, warnings, layout, holo). holo, when asked for: (foil.json record, {file name: bytes} as
    foil.files writes them, {"base" | "reverse": white plate PNG at card size, for the preview})."""
    def __new__(cls, png, notes, layout, holo=None):
        return super().__new__(cls, (png, notes, layout, holo))


# ---------------------------------------------------------------- cards
def layout_of(card_id):
    if not edit.valid_card_id(card_id):
        raise UserError(f"{card_id!r} is not a TCGdex card id (like swsh6-99 or sm10-19)")
    try:
        data = tcg.card(card_id)
    except (ValueError, cache.DownloadError) as e:
        raise UserError(str(e)) from e
    if data.get("category") == "Trainer":       # a Scarlet & Violet Trainer too: set again on its own kind's frame
        return "trainer"
    if modern.is_modern(card_id):
        return "modern"
    return "ex" if data.get("name", "").endswith(" V") else "me"


def read(card_id):
    layout = layout_of(card_id)
    reader = {"ex": compose.read_card, "me": mecompose.read_card, "modern": modern.read_card,
              "trainer": trainer.read_card}[layout]
    try:
        card = reader(card_id)
    except SystemExit as e:                     # the composers refuse unsupported cards this way
        raise UserError(str(e)) from e
    except cache.DownloadError as e:
        raise UserError(str(e)) from e
    return layout, card


def card_summary(card_id):
    layout, card = read(card_id)
    return {"id": card_id, "layout": layout, "name": f"{card['prefix']} {card['name']}".strip(),
            "type": card.get("type", card.get("kind")), "gen9": modern.is_modern(card_id), "card": edit.editable(card)}


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
    if key in _renders:
        _renders.move_to_end(key)
        return _renders[key]
    with _render_lock, warnings.catch_warnings(record=True) as caught, layers.recording() as drawn:
        warnings.simplefilter("always")
        layout, original = read(card_id)        # its warnings shown too (a Trainer's kind guessed, its text missing)
        if holo and layout == "trainer":
            raise UserError("holofoil is made for Pokemon cards only: turn it off for Trainers")
        card = edit.apply(original, edits) if edits else original
        if layout == "trainer":
            img, _ = trainer.compose_card(card_id, card=card)
            built = None
        elif layout == "modern":
            img, report = modern.compose_card(card_id, card, original)
            built = foil.build_modern(card_id, card, img, drawn, original) if holo else None
        else:
            if layout == "ex":
                img, _ = compose.compose(card_id, compose.Options(style=style, attacks=bool(attacks)), card=card)
            else:
                img, _ = mecompose.compose_card(card_id, bool(attacks), card=card)
            built = foil.build(card_id, layout, img, drawn, card) if holo else None
    notes = [str(w.message) for w in caught if "torch.jit" not in str(w.message)]
    made = None
    if built:
        record, written = foil.files(built, f"{card_id}-{layout}", card_id)
        plates = {k: _png(foil.card_space(built[k][1]), "L") for k in ("base", "reverse") if built[k]}
        made = (record, written, plates, hashlib.sha256(key.encode()).hexdigest()[:20])
    result = Render(_png(img), notes, layout, made)
    _renders[key] = result
    while len(_renders) > _RENDER_CACHE:
        _renders.popitem(last=False)
    return result


LAYOUT_NOTES = {"ex": "Restyled as a Scarlet & Violet ex card", "me": "Restyled as a Mega Evolution card",
                "trainer": "Restyled as a Scarlet & Violet Trainer"}


def card_changes(card_id, edits=None, style="hybrid", attacks=True):
    """-> (printed name, [what differs from the printed card]): the restyle itself, then edit.changes."""
    layout, original = read(card_id)
    card = edit.apply(original, edits) if edits else original
    notes = []
    if layout in LAYOUT_NOTES and not modern.is_modern(card_id):     # a Gen 9 Trainer is set again, not restyled
        notes.append(LAYOUT_NOTES[layout] + (" (translucent type panel)" if layout == "ex" and style == "panel" else ""))
        if not attacks and layout != "trainer":
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


def original_png(card_id):
    if not edit.valid_card_id(card_id):
        raise UserError(f"{card_id!r} is not a TCGdex card id")
    with _render_lock:                   # the first scan of a card is written to the cache
        try:
            return _png(tcg.scan(card_id), "RGBA")
        except (ValueError, cache.DownloadError) as e:
            raise UserError(str(e)) from e


def holo_json(made):
    """The foil part of /api/render's answer: the front's URL and per printing its plate (base64 PNG, card size),
    shader, material, cast-and-cure pattern and texture URLs (for the page's TCG Live shader preview), or null."""
    record, _, plates, token = made
    url = lambda name: f"/api/foil/file/{token}/{name}" if name else None
    out = {"front": url(record["front"])}
    for k in ("base", "reverse"):
        r = record[k]
        out[k] = None if not r else {
            "png": base64.b64encode(plates[k]).decode(), "shader": r["shader"], "material": r["material"],
            "cc": r.get("cc"), "textures": {t: url(r.get(t)) for t in ("wp", "etch", "cf")}}
    return out


def foil_file(token, name):
    """A texture of a recent render's holofoil, by the token its foil answer carried."""
    if not FILE_NAME.fullmatch(name or ""):
        raise UserError("bad texture name")
    for result in reversed(_renders.values()):
        made = result[3]
        if made and made[3] == token and name in made[1]:
            return made[1][name]
    raise LookupError(f"{name}: not among the recent renders (render the card again)")


def _extract_path(*parts):
    if not tcg.TCGLIVE:
        raise UserError("holofoil needs a Pokemon TCG Live extract (--tcglive DIR or SVRESTYLE_TCGLIVE)")
    return Path(tcg.TCGLIVE, *parts)


@functools.lru_cache(maxsize=1)
def foil_shaders():
    path = _extract_path(*FOIL_DATA, "shaders.json")
    if not path.exists():
        raise LookupError(f"{path}: the extract has no WebApp foil shaders")
    return path.read_bytes()


@functools.lru_cache(maxsize=1)
def foil_materials():
    """The extract's materials.json with its texture URLs pointing at /api/foil/tex/."""
    path = _extract_path(*FOIL_DATA, "materials.json")
    if not path.exists():
        raise LookupError(f"{path}: the extract has no WebApp foil materials")
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, meta in data.get("textures", {}).items():
        meta["url"] = f"/api/foil/tex/{name}.png"
    return json.dumps(data).encode()


def foil_texture(name):
    """-> (bytes, content type) of a material texture named in materials.json."""
    if not TEXTURE_NAME.fullmatch(name or "") or name not in json.loads(foil_materials())["textures"]:
        raise UserError(f"unknown foil texture {name!r}")
    png = _extract_path(*FOIL_TEX, f"{name}.png")
    if png.exists():
        return png.read_bytes(), "image/png"
    webp = _extract_path(*FOIL_TEX_WEB, f"{name}.webp")
    if webp.exists():
        return webp.read_bytes(), "image/webp"
    raise LookupError(f"{name}: not in the extract")


def static_file(rel):
    """-> (bytes, content type) of a file under static/ (the page's scripts: three.js, the foil preview)."""
    path = (STATIC / rel).resolve()
    if STATIC.resolve() not in path.parents or path.suffix not in STATIC_TYPES or not path.is_file():
        raise LookupError(f"/static/{rel}: not found")
    return path.read_bytes(), STATIC_TYPES[path.suffix]


def icon_png(type_name):
    if type_name not in edit.ENERGY:
        raise UserError(f"unknown energy type {type_name!r}")
    (crop, alpha, *_), _ = meframe.icon(type_name)
    rgba = np.dstack([np.clip(crop, 0, 255), alpha * 255]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def import_set(data):
    """Cards of a ptcg-sim custom-set export that need restyling, with the layout each gets; a set.json this editor
    exported comes back with its edits and options too."""
    if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
        raise UserError("expected a custom-set export (a JSON object with a \"cards\" list)")
    if export.is_set_data(data):
        try:
            return export.import_set_data(data, _set_entry)
        except ValueError as e:
            raise UserError(str(e)) from e
    out, seen = [], set()
    for c in data["cards"]:
        cid = c.get("id") if isinstance(c, dict) else None
        if not edit.valid_card_id(cid) or cid in seen:
            continue
        seen.add(cid)
        out.append(_set_entry(cid, c.get("name", ""), c.get("supertype")))
    return {"name": data.get("name", ""), "cards": out}


def _set_entry(cid, name, supertype):
    """A set list's entry: the layout the card gets, or why it is skipped. supertype: the export's or TCGdex's."""
    entry = {"id": cid, "name": name}
    if supertype == "Trainer":
        entry.update(layout="trainer", gen9=modern.is_modern(cid))
    elif supertype not in (None, "Pokémon", "Pokemon"):
        entry["skip"] = "not a Pokémon or Trainer"
    elif modern.is_modern(cid):
        entry["layout"] = "modern"
    else:
        entry["layout"] = "ex" if str(name).endswith(" V") else "me"
    return entry


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
        elif isinstance(c, dict) and c.get("layout") == "trainer" and "gen9" not in c:
            c = {**c, "gen9": modern.is_modern(c.get("id"))}
        out.append(c)
    return out


def export_set(items, title, code):
    """The list as a ptcg-sim custom set (export.py), rendering any card not rendered yet."""
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    if len(items) > MAX_ZIP_ITEMS:
        raise UserError(f"items: at most {MAX_ZIP_ITEMS} per download")
    try:
        return export.export_zip(items, title, code, read=read, render=render, apply=edit.apply, changelog=changelog)
    except ValueError as e:
        raise UserError(str(e)) from e


def sim_sets_dir():
    """The simulator's custom-sets folder, from SVRESTYLE_PTCG_SIM: the simulator's checkout (its
    client/src/assets/custom-sets is used, made if missing) or that folder itself; None when unset or neither."""
    raw = os.environ.get("SVRESTYLE_PTCG_SIM", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if (path / "client" / "src").is_dir():
        return path.joinpath(*SIM_SETS)
    if path.name == SIM_SETS[-1] and path.is_dir():
        return path
    return None


def push_set(items, title, code, force=False):
    """The list written into the simulator's custom-sets folder (export.write_set_dir), which the running simulator
    reloads: only cards whose edits or options changed since the last push are rendered (all of them with force)."""
    root = sim_sets_dir()
    if root is None:
        raise UserError("no ptcg-sim folder: start the GUI with SVRESTYLE_PTCG_SIM set to the simulator's folder "
                        "(start-site.bat sets it when the simulator is in Downloads)")
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    with _push_lock:                      # two pages pushing at once would interleave their files
        try:
            return export.write_set_dir(items, title, code, root, read=read, render=render, apply=edit.apply,
                                        changelog=changelog, force=bool(force))
        except ValueError as e:
            raise UserError(str(e)) from e


def zip_renders(items, title=""):
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    if len(items) > MAX_ZIP_ITEMS:
        raise UserError(f"items: at most {MAX_ZIP_ITEMS} per download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items:
            if not isinstance(it, dict):
                raise UserError("items: expected objects")
            png, _, layout, made = render(it.get("id"), it.get("edits"), it.get("style", "hybrid"),
                                          it.get("attacks", True), bool(it.get("foil")))
            z.writestr(f"{it['id']}-{layout}.png", png)
            for name, data in (made[1].items() if made else ()):
                z.writestr(name, data)
        z.writestr("CHANGELOG.md", changelog(items, title))
    return buf.getvalue()


# ---------------------------------------------------------------- projects
_project_lock = threading.Lock()


def projects_dir():
    return cache.CACHE / "projects"


def project_file(name):
    if not isinstance(name, str) or not PROJECT_NAME.fullmatch(name.strip()):
        raise UserError("project name: 1-60 letters, digits, spaces or . , ' ( ) & + # -")
    slug = re.sub(r"[^\w-]+", "-", name.strip().lower()).strip("-")
    return projects_dir() / f"{slug}.json"


def list_projects():
    out = []
    for f in projects_dir().glob("*.json") if projects_dir().exists() else ():
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"name": p.get("name", f.stem), "saved": p.get("saved", ""),
                    "cards": len((p.get("data") or {}).get("cards") or [])})
    return {"projects": sorted(out, key=lambda p: p["saved"], reverse=True)}


def load_project(name):
    f = project_file(name)
    if not f.exists():
        raise UserError(f"no saved project {name.strip()!r}")
    p = json.loads(f.read_text(encoding="utf-8"))
    p.setdefault("rev", 0)
    if isinstance(p.get("data"), dict) and isinstance(p["data"].get("cards"), list):
        p["data"]["cards"] = _current_entries(p["data"]["cards"])
    return p


def _check_data(data):
    if not isinstance(data, dict) or not isinstance(data.get("cards", []), list) \
            or not isinstance(data.get("edits", {}), dict) or not isinstance(data.get("options", {}), dict):
        raise UserError("project data: an object with a \"cards\" list and \"edits\" and \"options\" objects")


def _rev(f):
    try:
        return int(json.loads(f.read_text(encoding="utf-8")).get("rev", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def _write(f, name, data, rev):
    """Write the project atomically (a half-written file would lose the work it was saving). Under _project_lock."""
    saved = datetime.datetime.now().isoformat(timespec="seconds")
    text = json.dumps({"name": name, "saved": saved, "rev": rev, "data": data}, ensure_ascii=False)
    if len(text.encode()) > MAX_PROJECT:
        raise UserError(f"project too large (over {MAX_PROJECT >> 20} MB)")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, f)
    return {"name": name, "saved": saved, "rev": rev}


def save_project(name, data, client=None):
    _check_data(data)
    f = project_file(name)
    with _project_lock:
        out = _write(f, name.strip(), data, _rev(f) + 1 if f.exists() else 1)
        HUB.publish(f.stem, "replaced", {"rev": out["rev"], "client": client})
    return out


def apply_patch(data, patch):
    """The project data with a page's changes made. patch: {cards?: {removed: [id], added: [{index, card}],
    order?: [id]}, edits?: {id: {field: value} | null}, options?: {id: object | null}, setName?, foil?, current?}.
    code: the set code for the ptcg-sim export (export.py).
    A card's edits change field by field, so two people editing different fields of one card both keep theirs; null
    drops a card's edits or options. order is the list as the page has it: the cards it names take the places those
    cards hold, in its order, and cards it does not name (someone else's, just added) keep theirs. The page's
    applyPatch does the same."""
    if not isinstance(patch, dict):
        raise UserError("patch: an object")
    out = json.loads(json.dumps(data))
    out.setdefault("cards", [])
    cards = patch.get("cards")
    if cards is not None:
        if not isinstance(cards, dict) or not isinstance(cards.get("removed", []), list) \
                or not isinstance(cards.get("added", []), list) or not isinstance(cards.get("order", []), list):
            raise UserError("patch cards: {removed: [id], added: [{index, card}], order: [id]}")
        gone = set(cards.get("removed", []))
        out["cards"] = [c for c in out["cards"] if c.get("id") not in gone]
        for add in cards.get("added", []):
            card = add.get("card") if isinstance(add, dict) else None
            if not isinstance(card, dict) or not isinstance(card.get("id"), str):
                raise UserError("patch cards added: {index, card: {id, ...}}")
            if any(c.get("id") == card["id"] for c in out["cards"]):
                continue
            index = add.get("index") if isinstance(add.get("index"), int) else len(out["cards"])
            out["cards"].insert(max(0, min(index, len(out["cards"]))), card)
        out["cards"] = reorder(out["cards"], cards.get("order", []))
    for key in ("edits", "options"):
        changes = patch.get(key)
        if changes is None:
            continue
        if not isinstance(changes, dict):
            raise UserError(f"patch {key}: an object by card id")
        held = out.setdefault(key, {})
        for card_id, value in changes.items():
            if value is None:
                held.pop(card_id, None)
            elif not isinstance(value, dict):
                raise UserError(f"patch {key} {card_id}: an object or null")
            elif key == "edits":
                held[card_id] = {**(held.get(card_id) or {}), **value}
            else:
                held[card_id] = value
    for key in ("setName", "foil", "current", "code"):
        if key in patch:
            out[key] = patch[key]
    return out


def reorder(cards, order):
    rank = {card_id: i for i, card_id in enumerate(order) if isinstance(card_id, str)}
    named = iter(sorted((c for c in cards if c.get("id") in rank), key=lambda c: rank[c["id"]]))
    return [next(named) if c.get("id") in rank else c for c in cards]


def patch_project(name, patch, client=None):
    f = project_file(name)
    with _project_lock:
        held = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"data": {"version": 1}}
        data = apply_patch(held.get("data") or {}, patch)
        _check_data(data)
        out = _write(f, name.strip(), data, int(held.get("rev", 0)) + 1)
        HUB.publish(f.stem, "patch", {"rev": out["rev"], "client": client, "patch": patch})
    return out


def delete_project(name, renamed=None, client=None):
    """renamed: the name the work was saved under instead, so the others with it open move there."""
    f = project_file(name)
    if renamed is not None:
        project_file(renamed)                    # checks the name
    with _project_lock:
        if not f.exists():
            raise UserError(f"no saved project {name.strip()!r}")
        f.unlink()
        if renamed is not None:
            HUB.publish(f.stem, "renamed", {"to": renamed.strip(), "client": client})
        else:
            HUB.publish(f.stem, "deleted", {"client": client})
    return {"deleted": name.strip()}


class Room:
    def __init__(self):
        self.seq = 0
        self.log = deque(maxlen=EVENT_LOG)    # (seq, event, data)
        self.viewers = {}                     # client -> time of its last poll


class Hub:
    """What happens to each open project (by project file stem), for the pages polling it."""

    def __init__(self):
        self._cond = threading.Condition()
        self._rooms = {}

    def _room(self, key):
        return self._rooms.setdefault(key, Room())

    def _append(self, room, event, data):
        room.seq += 1
        room.log.append((room.seq, event, data))
        self._cond.notify_all()

    def _viewers(self, room, client=None):
        """Count the pages polling, `client` now one of them; a change in the count is news too."""
        now = time.monotonic()
        before = len(room.viewers)
        for c, seen in list(room.viewers.items()):
            if now - seen > VIEWER_GONE:
                del room.viewers[c]
        if client is not None:
            room.viewers[client] = now
        if len(room.viewers) != before:
            self._append(room, "viewers", {"viewers": len(room.viewers)})
        return len(room.viewers)

    def publish(self, key, event, data):
        with self._cond:
            self._append(self._room(key), event, data)

    def leave(self, key, client):
        with self._cond:
            room = self._room(key)
            if room.viewers.pop(client, None) is not None:
                self._append(room, "viewers", {"viewers": len(room.viewers)})

    def poll(self, key, after, client, rev, wait=POLL_WAIT):
        """-> {seq, viewers, events} once there are events past `after` or `wait` s pass; with hello {rev: rev()}
        at once when `after` is not a point this log can go on from (a new page, a restarted server, a page that
        was away longer than the log is long)."""
        deadline = time.monotonic() + wait
        with self._cond:
            room = self._room(key)
            self._viewers(room, client)
            oldest = room.log[0][0] if room.log else room.seq + 1
            if after < 0 or after > room.seq or after < oldest - 1:
                return {"seq": room.seq, "viewers": len(room.viewers), "events": [], "hello": {"rev": rev()}}
            while True:
                news = [{"event": e, "data": d} for n, e, d in room.log if n > after]
                left = deadline - time.monotonic()
                if news or left <= 0:
                    break
                self._cond.wait(left)
            room.viewers[client] = time.monotonic()
            return {"seq": room.seq, "viewers": len(room.viewers), "events": news}


HUB = Hub()


def poll_project(name, after, client):
    if not isinstance(client, str) or not 1 <= len(client) <= 80:
        raise UserError("poll: a client id")
    try:
        after = int(after)
    except (TypeError, ValueError):
        raise UserError("poll: after=SEQ") from None
    f = project_file(name)
    return HUB.poll(f.stem, after, client, lambda: _rev(f) if f.exists() else 0)


def leave_project(name, client):
    HUB.leave(project_file(name).stem, client)
    return {"left": name.strip()}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "svrestyle-gui"

    def log_message(self, fmt, *args):          # quiet: one line per error only
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise UserError("request too large")
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            raise UserError(f"invalid JSON: {e}") from e

    def _guard(self, fn):
        try:
            fn()
        except UserError as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except ValueError as e:                  # edit.apply's messages
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except LookupError as e:
            self._json({"error": str(e)}, HTTPStatus.NOT_FOUND)
        except Exception as e:                   # noqa: BLE001 - report, keep serving
            self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        url = urlparse(self.path)

        def go():
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._json({"energy": list(edit.ENERGY), "styles": list(STYLES),
                            "terms": [term for term, _ in wording.TERMS], "suggest": wording.available(),
                            "limits": {"abilities": edit.MAX_ABILITIES, "attacks": edit.MAX_ATTACKS,
                                       "cost": edit.MAX_COST, "retreat": edit.MAX_RETREAT},
                            "weakness": list(edit.WEAKNESS_VALUES), "resistance": list(edit.RESISTANCE_VALUES),
                            "foil": bool(tcg.TCGLIVE), "ptcgSim": sim_sets_dir() is not None})
            elif url.path == "/api/card":
                self._json(card_summary((parse_qs(url.query).get("id") or [""])[0].strip()))
            elif url.path == "/api/projects":
                name = (parse_qs(url.query).get("name") or [None])[0]
                self._json(load_project(name) if name is not None else list_projects())
            elif url.path == "/api/projects/poll":
                q = parse_qs(url.query)
                self._json(poll_project((q.get("name") or [""])[0], (q.get("after") or ["-1"])[0],
                                        (q.get("client") or [""])[0]))
            elif url.path.startswith("/api/original/") and url.path.endswith(".png"):
                self._send(HTTPStatus.OK, original_png(url.path[len("/api/original/"):-4]), "image/png")
            elif url.path.startswith("/api/icon/") and url.path.endswith(".png"):
                self._send(HTTPStatus.OK, icon_png(url.path[len("/api/icon/"):-4]), "image/png")
            elif url.path.startswith("/api/foil/file/"):
                token, _, name = url.path[len("/api/foil/file/"):].partition("/")
                self._send(HTTPStatus.OK, foil_file(token, name), "image/png")
            elif url.path == "/api/foil/shaders.json":
                self._send(HTTPStatus.OK, foil_shaders(), "application/json")
            elif url.path == "/api/foil/materials.json":
                self._send(HTTPStatus.OK, foil_materials(), "application/json")
            elif url.path.startswith("/api/foil/tex/") and url.path.endswith(".png"):
                self._send(HTTPStatus.OK, *foil_texture(url.path[len("/api/foil/tex/"):-4]))
            elif url.path.startswith("/static/"):
                self._send(HTTPStatus.OK, *static_file(url.path[len("/static/"):]))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        self._guard(go)

    def do_POST(self):
        url = urlparse(self.path)

        def go():
            body = self._body()
            if url.path == "/api/render":
                png, notes, layout, made = render(body.get("id"), body.get("edits"), body.get("style", "hybrid"),
                                                  body.get("attacks", True), bool(body.get("foil")))
                out = {"png": base64.b64encode(png).decode(), "warnings": notes, "layout": layout,
                       "changes": card_changes(body.get("id"), body.get("edits"), body.get("style", "hybrid"),
                                               body.get("attacks", True))[1]}
                if made:
                    out["foil"] = holo_json(made)
                self._json(out)
            elif url.path == "/api/set":
                self._json(import_set(body))
            elif url.path == "/api/suggest":
                try:
                    self._json(wording.suggest(body.get("text"), body.get("kind", "attack"), body.get("card")))
                except wording.SuggestionError as e:
                    self._json({"error": str(e)}, HTTPStatus.BAD_GATEWAY)
            elif url.path == "/api/zip":
                self._send(HTTPStatus.OK, zip_renders(body.get("items"), body.get("title", "")), "application/zip")
            elif url.path == "/api/export-set":
                self._send(HTTPStatus.OK, export_set(body.get("items"), body.get("title", ""), body.get("code")),
                           "application/zip")
            elif url.path == "/api/push-set":
                self._json(push_set(body.get("items"), body.get("title", ""), body.get("code"), body.get("force")))
            elif url.path == "/api/changelog":
                self._send(HTTPStatus.OK, changelog(body.get("items"), body.get("title", "")).encode(),
                           "text/markdown; charset=utf-8")
            elif url.path == "/api/projects":
                self._json(save_project(body.get("name"), body.get("data"), body.get("client")))
            elif url.path == "/api/projects/patch":
                self._json(patch_project(body.get("name"), body.get("patch"), body.get("client")))
            elif url.path == "/api/projects/delete":
                self._json(delete_project(body.get("name"), body.get("renamed"), body.get("client")))
            elif url.path == "/api/projects/leave":
                self._json(leave_project(body.get("name"), body.get("client")))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        self._guard(go)


def make_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port=8765, open_browser=True):
    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"svrestyle GUI at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
