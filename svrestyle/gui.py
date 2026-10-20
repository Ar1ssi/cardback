"""Local web GUI: load a card, edit its name, HP, abilities, attacks (costs, damage, text), weakness, resistance and
retreat, and render it in the ex or Mega Evolution layout; or import a custom-set export and restyle every older
card in it. Standard library only; serves on 127.0.0.1.

  python -m svrestyle gui [--port 8765] [--fonts DIR] [--no-browser]

API (JSON in and out unless noted):
  GET  /                      the page (gui.html)
  GET  /api/meta              energy types, limits
  GET  /api/card?id=ID        {id, layout: "ex"|"me", name, card: editable fields}
  POST /api/render            {id, edits?, style?, attacks?} -> {png: base64, warnings, layout}
  GET  /api/icon/TYPE.png     the energy icon used on cards (PNG)
  POST /api/set               a ptcg-sim custom-set export -> {cards: [{id, name, layout | skip}]}
  POST /api/zip               {items: [{id, edits?, style?, attacks?}]} -> application/zip
"""
import base64
import io
import json
import threading
import warnings
import webbrowser
import zipfile
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from . import cache, compose, edit, frame, mecompose, tcg

PAGE = Path(__file__).with_name("gui.html")
MAX_BODY = 8 << 20
MAX_ZIP_ITEMS = 60
CURRENT_ERAS = ("sv", "me")          # already Scarlet & Violet / Mega Evolution: nothing to restyle
STYLES = ("hybrid", "panel")

_render_lock = threading.Lock()       # LaMa, fonts and the asset builders are not thread-safe
_renders = OrderedDict()              # request key -> (png bytes, warnings); the last few, for zip downloads
_RENDER_CACHE = 48


class UserError(ValueError):
    """A problem with the request, shown to the user as is."""


# ---------------------------------------------------------------- cards
def layout_of(card_id):
    if not edit.valid_card_id(card_id):
        raise UserError(f"{card_id!r} is not a TCGdex card id (like swsh6-99 or sm10-19)")
    if card_id.startswith(CURRENT_ERAS):
        raise UserError(f"{card_id} is already a Scarlet & Violet / Mega Evolution card")
    try:
        data = tcg.card(card_id)
    except (ValueError, cache.DownloadError) as e:
        raise UserError(str(e)) from e
    return "ex" if data.get("name", "").endswith(" V") else "me"


def read(card_id):
    layout = layout_of(card_id)
    try:
        card = compose.read_card(card_id) if layout == "ex" else mecompose.read_card(card_id)
    except SystemExit as e:                     # the composers refuse unsupported cards this way
        raise UserError(str(e)) from e
    except cache.DownloadError as e:
        raise UserError(str(e)) from e
    return layout, card


def card_summary(card_id):
    layout, card = read(card_id)
    return {"id": card_id, "layout": layout, "name": f"{card['prefix']} {card['name']}".strip(),
            "type": card["type"], "card": edit.editable(card)}


def render(card_id, edits=None, style="hybrid", attacks=True):
    """-> (png bytes, warnings, layout); cached by request."""
    if style not in STYLES:
        raise UserError(f"style: one of {', '.join(STYLES)}")
    key = json.dumps([card_id, edits, style, bool(attacks)], sort_keys=True, ensure_ascii=False)
    if key in _renders:
        _renders.move_to_end(key)
        png, notes, layout = _renders[key]
        return png, notes, layout
    layout, card = read(card_id)
    if edits:
        card = edit.apply(card, edits)
    with _render_lock, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if layout == "ex":
            img, _ = compose.compose(card_id, compose.Options(style=style, attacks=bool(attacks)), card=card)
        else:
            img, _ = mecompose.compose_card(card_id, bool(attacks), card=card)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "PNG")
    notes = [str(w.message) for w in caught if "torch.jit" not in str(w.message)]
    _renders[key] = (buf.getvalue(), notes, layout)
    while len(_renders) > _RENDER_CACHE:
        _renders.popitem(last=False)
    return buf.getvalue(), notes, layout


def icon_png(type_name):
    if type_name not in edit.ENERGY:
        raise UserError(f"unknown energy type {type_name!r}")
    crop, alpha, *_ = frame.profile(type_name)["icon"]
    rgba = np.dstack([np.clip(crop, 0, 255), alpha * 255]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def import_set(data):
    """Cards of a ptcg-sim custom-set export that need restyling, with the layout each gets."""
    if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
        raise UserError("expected a custom-set export (a JSON object with a \"cards\" list)")
    out, seen = [], set()
    for c in data["cards"]:
        cid = c.get("id") if isinstance(c, dict) else None
        if not edit.valid_card_id(cid) or cid in seen:
            continue
        seen.add(cid)
        entry = {"id": cid, "name": c.get("name", "")}
        if cid.startswith(CURRENT_ERAS):
            entry["skip"] = "already current"
        elif c.get("supertype") not in (None, "Pokémon", "Pokemon"):
            entry["skip"] = "not a Pokémon"
        else:
            entry["layout"] = "ex" if str(c.get("name", "")).endswith(" V") else "me"
        out.append(entry)
    return {"name": data.get("name", ""), "cards": out}


def zip_renders(items):
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    if len(items) > MAX_ZIP_ITEMS:
        raise UserError(f"items: at most {MAX_ZIP_ITEMS} per download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items:
            if not isinstance(it, dict):
                raise UserError("items: expected objects")
            png, _, layout = render(it.get("id"), it.get("edits"), it.get("style", "hybrid"), it.get("attacks", True))
            z.writestr(f"{it['id']}-{layout}.png", png)
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
        except Exception as e:                   # noqa: BLE001 - report, keep serving
            self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        url = urlparse(self.path)

        def go():
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/meta":
                self._json({"energy": list(edit.ENERGY), "styles": list(STYLES),
                            "limits": {"abilities": edit.MAX_ABILITIES, "attacks": edit.MAX_ATTACKS,
                                       "cost": edit.MAX_COST, "retreat": edit.MAX_RETREAT}})
            elif url.path == "/api/card":
                self._json(card_summary((parse_qs(url.query).get("id") or [""])[0].strip()))
            elif url.path.startswith("/api/icon/") and url.path.endswith(".png"):
                self._send(HTTPStatus.OK, icon_png(url.path[len("/api/icon/"):-4]), "image/png")
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        self._guard(go)

    def do_POST(self):
        url = urlparse(self.path)

        def go():
            body = self._body()
            if url.path == "/api/render":
                png, notes, layout = render(body.get("id"), body.get("edits"), body.get("style", "hybrid"),
                                            body.get("attacks", True))
                self._json({"png": base64.b64encode(png).decode(), "warnings": notes, "layout": layout})
            elif url.path == "/api/set":
                self._json(import_set(body))
            elif url.path == "/api/zip":
                self._send(HTTPStatus.OK, zip_renders(body.get("items")), "application/zip")
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
