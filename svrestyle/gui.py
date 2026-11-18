"""Local web GUI: load a card, edit its name, HP, abilities, attacks (costs, damage, text), weakness, resistance and
retreat, and render it in the ex or Mega Evolution layout (Scarlet & Violet and Mega Evolution cards: edited in
place, see modern.py), with its TCG Live holofoil re-made to fit the edits; or load a Trainer (Item, Supporter,
Stadium, Pokemon Tool), edit its name, type and text, and render it as a Scarlet & Violet Trainer (trainer.py;
a Scarlet & Violet or Mega Evolution Trainer keeps its art and is set again on its kind's frame); or load an Energy card (older Basic or Special Energy is set on a Mega Evolution Energy frame, a Special one with its
name, text, the Energy it provides and frame to edit; Scarlet & Violet / Mega Evolution Energy is kept as printed,
energy.py); or import a custom-set export and restyle or edit every Pokemon, Trainer and Energy card in it. Standard library
only; serves on 127.0.0.1.

  python -m svrestyle gui [--port 8765] [--fonts DIR] [--tcglive DIR] [--ptcg-sim DIR] [--no-browser]

API (JSON in and out unless noted):
  GET  /                      the page (gui.html)
  GET  /api/meta              energy types, limits, weakness/resistance values, whether holofoil is available,
                              the wording terms the page capitalises as you type, whether /api/suggest works,
                              whether a ptcg-sim folder is set for /api/push-set (ptcgSim), whether
                              /api/publish-set can upload to the site's bucket (ptcgSite), whether a publish
                              key (SVRESTYLE_PUBLISH_TOKEN, sent as X-Svrestyle-Key) would unlock both from a
                              browser elsewhere (publishKey)
  POST /api/suggest           {text, kind: "attack"|"ability"|"trainer", card?: {name, type, stage, hp}} ->
                              {text, notes: [str]}: the text rewritten in printed-card wording by Claude
                              (wording.suggest: ANTHROPIC_API_KEY, else the Claude Code CLI's login)
  GET  /api/card?id=ID        {id, layout: "ex"|"me"|"modern"|"trainer"|"energy", name, card: editable fields}
  GET  /api/find/options      {sets: [{id, name}] newest first, rarities: [str]}: what the card search filters by
  POST /api/find              {name?, filters?: {category|types|stage|suffix|trainerType|regulationMark|rarity|set:
                              [str], hpMin?, hpMax?}, offset?, limit?} -> {total, cards: [{id, name, localId, set,
                              thumb}]}: a TCGdex search (tcg.find), Pokemon and Trainer cards, newest set first
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
  POST /api/follow            {id, edits} -> {abilities, attacks, weakness, resistance}: the card's costs, effect
                              text, weakness and resistance following its edited type (follow.follow), as edits
  POST /api/set               a ptcg-sim custom-set export ({cards: [{id, name, supertype}]}) -> {name, cards:
                              [{id, name, layout | skip}]}; or a set.json this editor exported (export.py) ->
                              also {code, edits, options, foil}: the cards with their edits restored
  POST /api/export-set        {items: [{id, edits?, style?, attacks?, foil?}], title?, code} -> application/zip:
                              the set for ptcg-sim (<set id>/set.json, <nnn>/high.webp + low.webp, CHANGELOG.md;
                              see export.py)
  POST /api/push-set          {items, title?, code, force?, allowShrink?} -> a job (GET /api/set-job): the same set
                              written into the simulator's custom-sets folder (SVRESTYLE_PTCG_SIM), which the
                              running simulator reloads; only cards changed since the last push are rendered
                              (export.write_set_dir), every card with force. A push never takes a card out of the
                              set: the cards it holds that the list does not name stay (setjobs.keep_unlisted).
                              Only cards that cannot be sent again (a set this editor did not write) would be lost:
                              refused, 409 {error, drops: {setId, code, incoming, copies, names, message}}, unless
                              allowShrink: the page asks, then sends again with it
  POST /api/publish-set       {items, title?, code, force?, allowShrink?} -> a job: the set pushed as above, then
                              uploaded to the site's bucket by the simulator's scripts/custom-sets/publish.mjs
                              (rclone remote CUSTOM_SETS_R2_REMOTE); the site (Render) picks it up within ~10 s.
                              Fewer cards than the site's copy is refused like fewer than the simulator's
  GET  /api/set-job?job=ID    a send or publish job: {id, kind: "push"|"publish", code, setId, state: "running"|"done"|
                              "failed"|"cancelled", phase: "rendering"|"writing"|"uploading", done, total, current,
                              elapsed (s), cancelling, result, error, pushed}. result (state done) is the push's
                              answer {setId, rendered, reused, written: [path], unchanged, removed: [card folder]},
                              plus published: {sets: [{setId, uploaded, uploadedBytes, reused, skipped, unchanged}],
                              deleted} for a publish. pushed is that answer once the files are in the simulator's
                              folder (a publish that then failed or was cancelled has it). Jobs run in the
                              background (a render takes 10-20 s a card, an upload minutes, tunnels cut a request at
                              ~100 s): the page polls. Asking to start the job a running one is doing follows it
  POST /api/push-plan         {code} -> what pushing every saved project, merged, as that set would change in the
                              simulator's folder, without rendering: {setId, code, title, projects, repeats, incoming,
                              existing, added: [{id, name, lines}], changed: [{id, name, lines}], removed: [{id, name}],
                              moved, unchanged, unreadable: [{name, why}], notes: [str], items}. items is the list to
                              send to /api/push-set or /api/publish-set to push exactly this ("Push all"); cards already
                              in the set keep their numbers. unreadable: saved projects that could not be read (their
                              cards are in none of the lists); notes: what the merge left off (holofoil, no extract)
  POST /api/set-job/cancel    {job} -> its progress (cancelling: true). Before the first file is written nothing
                              changes; a publish stops its upload (the site keeps the copy it had); a send that is
                              writing finishes. The three /api/set-job routes need what push-set needs: this
                              machine's browser, or the publish key
  POST /api/zip              {items: [{id, edits?, style?, attacks?, foil?}], title?} -> application/zip (with foil
                              also the 1024x1024 front, the white plates and <card>-foil.json, see foil.files), and
                              CHANGELOG.md
  POST /api/changelog         {items, title?} -> text/markdown: every card's changes from its printed card
  GET  /api/projects          {projects: [{name, saved, cards}]}, newest first
  GET  /api/projects?name=N   {name, saved, rev, data}: a saved project (the page's card list, edits and options)
  POST /api/projects          {name, data, client?} -> {name, saved, rev}: save (overwrites a project of that name)
  POST /api/projects/patch    {name, patch, client?} -> {name, saved, rev}: change only what the patch names (see
                              projects.apply_patch), making the project if there is none
  POST /api/projects/delete   {name, renamed?, client?} -> {deleted: name}
  GET  /api/projects/poll?name=N&after=SEQ&client=C  long poll for what happens to the project: answers when there
                              is news past SEQ (or after projects.POLL_WAIT s with none) -> {seq, viewers, events:
                              [{event, data}]}: patch {rev, client, patch}, replaced {rev, client}, renamed {to,
                              client}, deleted {client}, viewers {viewers}. after=-1 (or a SEQ it no longer has) ->
                              {seq, viewers, events: [], hello: {rev}} at once: the page reads the project again if
                              its rev differs. (Long polls, not an event stream: Cloudflare quick tunnels hold
                              streams.)
  POST /api/projects/merged   {} -> {id, state: "running"|"done"|"failed"|"cancelled", done, total, current, projects, repeats,
                              failed: [{id, why}], unreadable: [{name, why}], notes: [str], error}: start building
                              every saved project (unreadable: those that could not be read), merged into one list
                              (a card in several: the newest project's copy), as one zip in the background; a build
                              already running is joined. Poll with GET /api/projects/merged?job=ID (same answer)
  POST /api/projects/merged/cancel  {job} -> its progress (cancelling: true): the build stops after the card it is
                              rendering, state "cancelled", and its zip is removed
  GET  /api/projects/merged/file?job=ID  the finished zip (application/zip, as a download): the renders,
                              CHANGELOG.md, and NOT-INCLUDED.txt for cards that could not be rendered, projects that
                              could not be read and holofoil left off
  POST /api/projects/leave    {name, client}: the page is closing (it stops counting as a viewer at once)
  GET  /api/projects/history?name=N  {name, versions: [{id, saved, rev, cards}]}: the project's kept versions,
                              newest first
  POST /api/projects/restore  {name, version, client?} -> {name, saved, rev}: the project as it was in that version
                              (what it is now is kept as a version first)
  GET  /api/projects/trash    {trash: [{id, name, deleted, saved, cards}]}: deleted projects, newest first
  POST /api/projects/undelete {id} -> {name, saved, rev}: a deleted project back, under its name (or "<name>
                              (restored)" when another project has it now)

This module is the server and what the page asks of a card; the work behind the routes is in rendering.py (a
card read and rendered), projects.py (saved projects, their history and trash, the long polls), setjobs.py (sends,
publishes and the push-all plan) and merged.py (every project as one download).
"""
import base64
import functools
import hmac
import io
import json
import os
import re
import shutil
import threading
import traceback
import warnings
import webbrowser
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import cache, edit, export, follow, meframe, merged, modern, projects, rendering, setjobs, tcg, wording
from .rendering import UserError

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
LOOPBACK = ("127.0.0.1", "localhost", "::1")
PUBLISH_KEY_HEADER = "X-Svrestyle-Key"  # a custom header: a page elsewhere cannot send it without a CORS preflight
MIN_PUBLISH_KEY = 16                  # characters; a shorter SVRESTYLE_PUBLISH_TOKEN is ignored, it could be guessed


# ---------------------------------------------------------------- what the page asks of a card
def find_cards(body):
    """A card search on TCGdex (tcg.find): {"name", "filters", "offset", "limit"} -> {"total", "cards"}."""
    try:
        return tcg.find(body.get("name"), body.get("filters"), body.get("offset", 0), body.get("limit", 48))
    except (ValueError, TypeError, cache.DownloadError) as e:
        raise UserError(str(e)) from e


def find_options():
    """What the search offers to filter by, from TCGdex: the sets (newest first) and the rarities."""
    try:
        return {"sets": tcg.sets(), "rarities": tcg.rarities()}
    except cache.DownloadError as e:
        raise UserError(str(e)) from e


def card_summary(card_id):
    layout, card = rendering.read(card_id)
    return {"id": card_id, "layout": layout, "name": f"{card['prefix']} {card['name']}".strip(),
            "type": card.get("type", card.get("kind")), "gen9": modern.is_modern(card_id),
            "energy": card.get("energy"),
            "ace": bool(card.get("ace")), "card": edit.editable(card)}


def follow_type(card_id, edits):
    """What goes with the card's edited type (follow.follow): its costs, effect text, weakness and resistance, as
    edits for the page to take."""
    layout, original = rendering.read(card_id)
    if layout in ("trainer", "energy") or not edit.retypes(original):
        raise UserError("only a Pokémon that can change type has a type to follow")
    return follow.follow(original, edit.apply(original, edits) if edits else original)


def original_png(card_id):
    if not edit.valid_card_id(card_id):
        raise UserError(f"{card_id!r} is not a TCGdex card id")
    with rendering._render_lock:                   # the first scan of a card is written to the cache
        try:
            return rendering._png(tcg.scan(card_id), "RGBA")
        except (ValueError, cache.DownloadError) as e:
            raise UserError(str(e)) from e


def holo_json(made):
    """The foil part of /api/render's answer: the front's URL and per printing its plate (base64 PNG, card size),
    shader, material, cast-and-cure pattern and texture URLs (for the page's TCG Live shader preview), or null."""
    url = lambda name: f"/api/foil/file/{made.token}/{name}" if name else None
    out = {"front": url(made.record["front"])}
    for k in ("base", "reverse"):
        r = made.record[k]
        out[k] = None if not r else {
            "png": base64.b64encode(made.plates[k]).decode(), "shader": r["shader"], "material": r["material"],
            "cc": r.get("cc"), "textures": {t: url(r.get(t)) for t in ("wp", "etch", "cf")}}
    return out


def foil_file(token, name):
    """A texture of a recent render's holofoil, by the token its foil answer carried."""
    if not FILE_NAME.fullmatch(name or ""):
        raise UserError("bad texture name")
    with rendering._renders_lock:
        recent = list(rendering._renders.values())
    for result in reversed(recent):
        made = result.holo
        if made and made.token == token and name in made.files:
            return made.files[name]
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
            return export.import_set_data(data, rendering._set_entry)
        except ValueError as e:
            raise UserError(str(e)) from e
    out, seen = [], set()
    for c in data["cards"]:
        cid = c.get("id") if isinstance(c, dict) else None
        if not edit.valid_card_id(cid) or cid in seen:
            continue
        seen.add(cid)
        out.append(rendering._set_entry(cid, c.get("name", ""), c.get("supertype")))
    return {"name": data.get("name", ""), "cards": out}


def export_set(items, title, code):
    """The list as a ptcg-sim custom set (export.py), rendering any card not rendered yet."""
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    if len(items) > MAX_ZIP_ITEMS:
        raise UserError(f"items: at most {MAX_ZIP_ITEMS} per download")
    try:
        return export.export_zip(items, title, code, read=rendering.read, render=rendering.render, apply=edit.apply,
                                 changelog=rendering.changelog, is_ace=rendering._is_ace)
    except ValueError as e:
        raise UserError(str(e)) from e


def publish_key():
    """SVRESTYLE_PUBLISH_TOKEN: the key that lets a browser outside this machine (through the tunnel) send and
    publish sets, or None when unset or shorter than MIN_PUBLISH_KEY."""
    key = os.environ.get("SVRESTYLE_PUBLISH_TOKEN", "").strip()
    if not key:
        return None
    if len(key) < MIN_PUBLISH_KEY:
        warnings.warn(f"SVRESTYLE_PUBLISH_TOKEN ignored: shorter than {MIN_PUBLISH_KEY} characters", stacklevel=2)
        return None
    return key


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
            png, _, layout, made = rendering.render(it.get("id"), it.get("edits"), it.get("style", "hybrid"),
                                                    it.get("attacks", True), bool(it.get("foil")))
            z.writestr(f"{it['id']}-{layout}.png", png)
            for name, data in (made.files.items() if made else ()):
                z.writestr(name, data)
        z.writestr("CHANGELOG.md", rendering.changelog(items, title))
    return buf.getvalue()


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

    def _file(self, path, filename):
        """Send a (large) file in chunks, as a download."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
        except (BrokenPipeError, ConnectionResetError):          # the page gave up on it
            pass

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

    def _from_this_machine(self):
        """Whether the request comes from a browser on this machine. The site is often shared through a Cloudflare
        tunnel (start-site.bat), whose requests carry Cf-Connecting-Ip and the tunnel's host name; a DNS-rebinding
        page carries its own host name; a page elsewhere posting here carries its own Origin."""
        if self.headers.get("Cf-Connecting-Ip") or self.headers.get("Cf-Ray"):
            return False
        host, origin = self.headers.get("Host"), self.headers.get("Origin")
        if not host or urlparse(f"//{host}").hostname not in LOOPBACK:
            return False
        return origin is None or urlparse(origin).hostname in LOOPBACK

    def _has_publish_key(self):
        """Whether the request carries SVRESTYLE_PUBLISH_TOKEN in PUBLISH_KEY_HEADER (compared in constant time)."""
        key, sent = publish_key(), self.headers.get(PUBLISH_KEY_HEADER)
        return bool(key and sent) and hmac.compare_digest(sent.strip().encode(), key.encode())

    def _trusted(self):
        """Whether the request may write outside the editor: this machine's browser, or one holding the key."""
        return self._from_this_machine() or self._has_publish_key()

    def _refused(self, what):
        """Answers 403 and returns True unless the request may write outside the editor (_trusted)."""
        if self._trusted():
            return False
        self._json({"error": f"{what} works only in a browser on the machine running the editor or with the publish "
                             "key"}, HTTPStatus.FORBIDDEN)
        return True

    def _can_push(self):
        """/api/push-set writes into the simulator checkout: only for a trusted request, and only when the checkout
        is known."""
        return self._trusted() and setjobs.sim_sets_dir() is not None

    def _can_publish(self):
        """/api/publish-set: as push-set, and the bucket remote and the simulator's publish script are known."""
        return self._can_push() and setjobs.site_remote() is not None and setjobs.sim_checkout() is not None

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
            traceback.print_exc()                # not the user's mistake: its trace for whoever runs the server
            self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        url = urlparse(self.path)

        def go():
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._json({"energy": list(edit.ENERGY), "styles": list(rendering.STYLES), "ptcgSim": self._can_push(),
                            "ptcgSite": self._can_publish(),
                            "publishKey": publish_key() is not None and setjobs.sim_sets_dir() is not None,
                            "terms": [term for term, _ in wording.TERMS], "suggest": wording.available(),
                            "limits": {"abilities": edit.MAX_ABILITIES, "attacks": edit.MAX_ATTACKS,
                                       "cost": edit.MAX_COST, "retreat": edit.MAX_RETREAT},
                            "weakness": list(edit.WEAKNESS_VALUES), "resistance": list(edit.RESISTANCE_VALUES),
                            "foil": bool(tcg.TCGLIVE)})
            elif url.path == "/api/find/options":
                self._json(find_options())
            elif url.path == "/api/card":
                self._json(card_summary((parse_qs(url.query).get("id") or [""])[0].strip()))
            elif url.path == "/api/projects":
                name = (parse_qs(url.query).get("name") or [None])[0]
                self._json(projects.load_project(name) if name is not None else projects.list_projects())
            elif url.path == "/api/set-job":
                if not self._refused("Following a send or publish"):
                    self._json(setjobs.set_job_progress((parse_qs(url.query).get("job") or [""])[0]))
            elif url.path == "/api/projects/merged":
                self._json(merged.merged_progress((parse_qs(url.query).get("job") or [""])[0]))
            elif url.path == "/api/projects/merged/file":
                self._file(merged.merged_file((parse_qs(url.query).get("job") or [""])[0]), "all-projects.zip")
            elif url.path == "/api/projects/poll":
                q = parse_qs(url.query)
                self._json(projects.poll_project((q.get("name") or [""])[0], (q.get("after") or ["-1"])[0],
                                                 (q.get("client") or [""])[0]))
            elif url.path == "/api/projects/history":
                self._json(projects.list_versions((parse_qs(url.query).get("name") or [""])[0]))
            elif url.path == "/api/projects/trash":
                self._json(projects.list_trash())
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
                png, notes, layout, made = rendering.render(body.get("id"), body.get("edits"),
                                                            body.get("style", "hybrid"), body.get("attacks", True),
                                                            bool(body.get("foil")))
                out = {"png": base64.b64encode(png).decode(), "warnings": notes, "layout": layout,
                       "changes": rendering.card_changes(body.get("id"), body.get("edits"),
                                                         body.get("style", "hybrid"), body.get("attacks", True))[1]}
                if made:
                    out["foil"] = holo_json(made)
                self._json(out)
            elif url.path == "/api/find":
                self._json(find_cards(body))
            elif url.path == "/api/follow":
                self._json(follow_type(body.get("id"), body.get("edits")))
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
            elif url.path in ("/api/push-set", "/api/publish-set"):
                kind = "publish" if url.path == "/api/publish-set" else "push"
                if self._refused("Publish to site" if kind == "publish" else "Send to ptcg-sim"):
                    return
                try:
                    self._json(setjobs.start_set_job(kind, body.get("items"), body.get("title", ""), body.get("code"),
                                                     body.get("force"), body.get("allowShrink") is True))
                except setjobs.DropsCards as e:
                    self._json({"error": str(e), "drops": e.info}, HTTPStatus.CONFLICT)
            elif url.path == "/api/set-job/cancel":
                if not self._refused("Cancelling a send or publish"):
                    self._json(setjobs.cancel_set_job(body.get("job")))
            elif url.path == "/api/push-plan":
                if not self._refused("Push all"):
                    self._json(setjobs.push_all_plan(body.get("code")))
            elif url.path == "/api/changelog":
                self._send(HTTPStatus.OK, rendering.changelog(body.get("items"), body.get("title", "")).encode(),
                           "text/markdown; charset=utf-8")
            elif url.path == "/api/projects":
                self._json(projects.save_project(body.get("name"), body.get("data"), body.get("client")))
            elif url.path == "/api/projects/merged":
                self._json(merged.start_merged_zip())
            elif url.path == "/api/projects/merged/cancel":
                self._json(merged.cancel_merged_zip(body.get("job")))
            elif url.path == "/api/projects/patch":
                self._json(projects.patch_project(body.get("name"), body.get("patch"), body.get("client")))
            elif url.path == "/api/projects/delete":
                self._json(projects.delete_project(body.get("name"), body.get("renamed"), body.get("client")))
            elif url.path == "/api/projects/leave":
                self._json(projects.leave_project(body.get("name"), body.get("client")))
            elif url.path == "/api/projects/restore":
                self._json(projects.restore_version(body.get("name"), body.get("version"), body.get("client")))
            elif url.path == "/api/projects/undelete":
                self._json(projects.undelete_project(body.get("id")))
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
