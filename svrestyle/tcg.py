"""Card data (TCGdex) and card scans.

Scans come from a local Pokemon TCG Live extract when SVRESTYLE_TCGLIVE points at one (the game's 1024x1024 card
fronts, see tcglive_front), else from pokemontcg.io (733x1024), with TCGdex's 600x825 as a last fallback.
"""
import hashlib
import json
import os
import re
import time
import warnings
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from . import cache, edit

TCGDEX = "https://api.tcgdex.net/v2/en"
W, H = 733, 1024   # every layout constant in this package is measured at this size

# Pokemon TCG Live extract: a folder with cardart/bundles/<set>_en_<NNN> (the game's card bundles) and/or
# WebApp/assets/cards/<set>_<N>.webp (the same fronts, re-encoded).
TCGLIVE = os.environ.get("SVRESTYLE_TCGLIVE")
# A TCG Live front is the 733x1024 card plus 145 px of print bleed left and right, at the pokemontcg.io scale
# (SIFT-matched against pokemontcg.io on Sun & Moon, Sword & Shield, Scarlet & Violet and Mega Evolution cards:
# scale 1.0000, offset 145.0 px, 0.0 px).
BLEED_X = 145
# TCGdex set id -> (TCG Live set, number offset). Other sets map by rule: sv03.5 -> sv3-5, me01 -> me1.
TCGLIVE_SETS = {
    "sma": ("sm11-5", 69),     # Hidden Fates Shiny Vault: SV1 is Hidden Fates 70
    "sm115": ("sm11-5", 0),
    "sm35": ("sm3-5", 0),
    "sm75": ("sm7-5", 0),
    "swshp": ("swshbsp", 0),
    "smp": ("smbsp", 0),
    "svp": ("svbsp", 0),
    "mep": ("mebsp", 0),
    "xyp": ("xybsp", 0),
    "bwp": ("bwbsp", 0),
}
# Sets released as a pair under one number: TCGdex suffixes the number (sv10.5b), pokemontcg.io and TCG Live prefix the
# series with the mascot's letter (zsv10pt5, zsv10-5: Zekrom's Black Bolt; Reshiram's White Flare).
PAIRED_SETS = {"sv10.5b": "z", "sv10.5w": "r"}
# Radiant Collections number apart from their set (bw11-RC20 is not bw11-20): no TCG Live id is known for them.
SUBSET_PREFIXES = ("RC",)
CORNER_REF = "sv03-135"   # its pokemontcg.io scan's alpha gives TCG Live fronts the printed card's rounded corners
# Card search. TCGdex also lists TCG Pocket (the phone game), whose sets keep their art under .../tcgp/.
POCKET = "/tcgp/"
POCKET_RARITIES = {"None", "Crown", "One Diamond", "Two Diamond", "Three Diamond", "Four Diamond", "One Star",
                   "Two Star", "Three Star", "One Shiny", "Two Shiny"}
# filter name -> TCGdex card field; each filter is a list of exact values, any of which matches
FIND_FIELDS = {"category": "category", "types": "types", "stage": "stage", "suffix": "suffix",
               "trainerType": "trainerType", "energyType": "energyType", "regulationMark": "regulationMark", "rarity": "rarity", "set": "set.id"}
FIND_CATEGORIES = ("Pokemon", "Trainer", "Energy")
FIND_MAX = 96                                 # most cards one search answer carries


def card(card_id):
    data = cache.fetch_json(f"{TCGDEX}/cards/{card_id}", cache.path("data", f"{card_id}.json"))
    if not data.get("id"):
        raise ValueError(f"TCGdex has no card {card_id!r}")
    return data


def set_info(set_id):
    return cache.fetch_json(f"{TCGDEX}/sets/{set_id}", cache.path("data", f"set-{set_id}.json"))


def search(field, value):
    """TCGdex card briefs whose `field` equals value (cached)."""
    value = value.replace("’", "'")          # TCGdex writes a straight apostrophe (Farfetch'd)
    slug = re.sub(r"[^\w-]+", "_", value)
    return cache.fetch_json(f"{TCGDEX}/cards?{field}=eq:{quote(value)}",
                            cache.path("data", f"search-{field}-{slug}.json")) or []


def _fresh_json(url, name, days=1):
    """cache.fetch_json that asks TCGdex again once the saved answer is `days` old (new sets and cards come out); the
    old answer stands in when TCGdex cannot be reached."""
    dest = cache.path("data", name)
    old = None
    if dest.exists() and time.time() - dest.stat().st_mtime > days * 86400:
        old = dest.read_bytes()
        dest.unlink()
    try:
        return cache.fetch_json(url, dest)
    except cache.DownloadError:
        if old is None:
            raise
        dest.write_bytes(old)
        return json.loads(old)


def _all_sets():
    return _fresh_json(f"{TCGDEX}/sets", "sets.json")


def _pocket(set_brief):
    return POCKET in set_brief.get("logo", "") + set_brief.get("symbol", "")


def sets():
    """The sets with real cards the editor takes, newest first (TCG Pocket's left out): [{"id", "name"}]."""
    return [{"id": s["id"], "name": s["name"]} for s in reversed(_all_sets())
            if not _pocket(s) and edit.valid_card_id(f"{s['id']}-1")]


def rarities():
    return [r for r in _fresh_json(f"{TCGDEX}/rarities", "rarities.json") if r not in POCKET_RARITIES]


def _natural(local_id):
    """Sort key for card numbers: 9 before 10, 099 before SV01."""
    return [int(t) if i % 2 else t for i, t in enumerate(re.split(r"(\d+)", local_id))]


def find(name="", filters=None, offset=0, limit=48):
    """Cards of a TCGdex search, newest set first: {"total": n, "cards": [one page of {id, name, localId, set, thumb}]}.
    name: part of the card's name. filters: FIND_FIELDS keys, each a list of exact values (any matches), and "hpMin" /
    "hpMax". Pokemon, Trainer and Energy cards only, TCG Pocket's left out, and the cards whose id the editor takes (edit.CARD_ID:
    not 30th-053, 2011bw-1)."""
    filters = filters or {}
    if not isinstance(filters, dict) or not isinstance(name or "", str):
        raise ValueError("search: a name and a dict of filters")
    name = (name or "").strip().replace("’", "'")
    query = [f"name={quote(name)}"] if name else []
    for key, field in FIND_FIELDS.items():
        values = filters.get(key) or []
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            raise ValueError(f"{key}: a list of names")
        if key == "category" and set(values) - set(FIND_CATEGORIES):
            raise ValueError(f"category: one of {', '.join(FIND_CATEGORIES)}")
        if values:
            query.append(f"{field}=eq:" + "%7C".join(quote(v, safe="") for v in values))
    for key, op in (("hpMin", "gte"), ("hpMax", "lte")):
        hp = filters.get(key)
        if hp is None:
            continue
        if not isinstance(hp, int) or isinstance(hp, bool) or not 0 <= hp <= 1000:
            raise ValueError(f"{key}: a number from 0 to 1000")
        query.append(f"hp={op}:{hp}")
    if not query:
        raise ValueError("Type a name or pick a filter")
    if not filters.get("category"):
        query.append("category=eq:" + "%7C".join(FIND_CATEGORIES))
    url = f"{TCGDEX}/cards?" + "&".join(query)
    briefs = _fresh_json(url, f"find-{hashlib.sha1(url.encode()).hexdigest()[:16]}.json") or []

    listed = _all_sets()
    order = {s["id"]: i for i, s in enumerate(listed)}
    names = {s["id"]: s["name"] for s in listed}
    pocket = {s["id"] for s in listed if _pocket(s)}
    hits = []
    for b in briefs:
        set_id = b["id"].rsplit("-", 1)[0]
        if edit.valid_card_id(b["id"]) and set_id not in pocket and POCKET not in b.get("image", ""):
            hits.append((-order.get(set_id, -1), _natural(b.get("localId", "")), b, set_id))
    hits.sort(key=lambda h: h[:2])
    offset, limit = max(int(offset), 0), min(max(int(limit), 1), FIND_MAX)
    return {"total": len(hits), "cards": [
        {"id": b["id"], "name": b.get("name", ""), "localId": b.get("localId", ""), "set": names.get(set_id, set_id),
         "thumb": f"{b['image']}/low.webp" if b.get("image") else ""}
        for _, _, b, set_id in hits[offset:offset + limit]]}


def _split(card_id):
    """TCGdex id -> (series, set number, ".5" or None, card number), with a paired set's letter on the series:
    sv03.5-065 -> ("sv", "3", ".5", "65"), sv10.5b-001 -> ("zsv", "10", ".5", "1")."""
    m = re.fullmatch(r"([a-z]+)0*(\d*)(\.5)?([a-z]?)-0*(\w+)", card_id)
    if not m:
        raise ValueError(f"unexpected card id {card_id!r}")
    prefix, num, half, pair, local = m.groups()
    if pair:
        set_id = card_id.split("-", 1)[0]
        if set_id not in PAIRED_SETS:
            raise ValueError(f"unexpected card id {card_id!r}")
        prefix = PAIRED_SETS[set_id] + prefix
    return prefix, num, half, local


def pokemontcg_id(card_id):
    """TCGdex id -> pokemontcg.io image path: swsh6-099 -> swsh6/99, sv03.5-065 -> sv3pt5/65, sma-SV33 -> sma/SV33,
    sv10.5b-001 -> zsv10pt5/1."""
    prefix, num, half, local = _split(card_id)
    return f"{prefix}{num}{'pt5' if half else ''}/{local}"


def tcglive_id(card_id):
    """TCGdex id -> TCG Live card id: swsh6-099 -> swsh6_99, sv03.5-065 -> sv3-5_65, sma-SV33 -> sm11-5_102,
    swshp-SWSH055 -> swshbsp_55, me01-054 -> me1_54, sv10.5w-001 -> rsv10-5_1."""
    prefix, num, half, local = _split(card_id)
    set_id, offset = TCGLIVE_SETS.get(f"{prefix}{num}", (f"{prefix}{num}{'-5' if half else ''}", 0))
    digits = re.fullmatch(r"[A-Za-z]*0*(\d+)", local)
    if not digits:
        raise ValueError(f"{card_id}: no card number")
    if local.upper().startswith(SUBSET_PREFIXES):
        raise ValueError(f"{card_id}: a sub-set card, numbered apart from its set (no TCG Live id known)")
    return f"{set_id}_{int(digits.group(1)) + offset}"


def tcglive_front(card_id, root=None):
    """The card's 1024x1024 TCG Live front as an RGB PIL image, or None if the extract does not have it.
    Decoded from the game bundle (needs UnityPy) when possible, else read from the web app's .webp copy."""
    root = root or TCGLIVE
    if not root:
        return None
    set_id, number = tcglive_id(card_id).rsplit("_", 1)
    bundle = Path(root, "cardart", "bundles", f"{set_id}_en_{int(number):03d}")
    if bundle.exists():
        try:
            import UnityPy
        except ImportError:
            UnityPy = None
        if UnityPy:
            for obj in UnityPy.load(str(bundle)).objects:
                if obj.type.name == "Texture2D" and obj.peek_name() == bundle.name:
                    return obj.read().image.convert("RGB")
    webp = Path(root, "WebApp", "assets", "cards", f"{set_id}_{number}.webp")
    if webp.exists():
        return Image.open(webp).convert("RGB")
    return None


def _web_scan(card_id, dest):
    try:
        cache.fetch(f"https://images.pokemontcg.io/{pokemontcg_id(card_id)}_hires.png", dest)
    except cache.DownloadError:
        base = card(card_id).get("image")
        if not base:
            raise
        warnings.warn(f"{card_id}: no 733x1024 scan, using TCGdex's 600x825 (softer result)")
        cache.fetch(f"{base}/high.png", dest)


def _open(dest):
    img = Image.open(dest).convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return img


def _tcglive_scan(card_id, dest):
    """Crop the TCG Live front to the card and give it rounded corners. False if the extract lacks the card."""
    try:
        live_id = tcglive_id(card_id)
    except ValueError as e:                   # xy10-111a, bw11-RC20: no TCG Live id to look for
        warnings.warn(f"{e}: using pokemontcg.io")
        return False
    front = tcglive_front(card_id)
    if front is None:
        warnings.warn(f"{card_id}: not in the TCG Live extract ({live_id}), using pokemontcg.io")
        return False
    if front.size != (1024, 1024):
        front = front.resize((1024, 1024), Image.LANCZOS)
    corners = cache.path("web-scans", f"{CORNER_REF}.png")
    if not corners.exists():
        _web_scan(CORNER_REF, corners)
    rgb = front.crop((BLEED_X, 0, BLEED_X + W, H))
    Image.merge("RGBA", (*rgb.split(), _open(corners).getchannel("A"))).save(dest)
    return True


def scan(card_id, expect_sha=None):
    """The card scan as float32 RGBA at 733x1024."""
    dest = cache.path("scans", f"{card_id}.png")
    if not dest.exists() and not (TCGLIVE and _tcglive_scan(card_id, dest)):
        _web_scan(card_id, dest)
        if expect_sha and not cache.sha256(dest).startswith(expect_sha):
            warnings.warn(f"{card_id}: scan differs from the one the layout was measured on; check the output")
    return np.asarray(_open(dest)).astype(np.float32)
