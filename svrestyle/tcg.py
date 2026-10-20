"""Card data (TCGdex) and card scans.

Scans come from a local Pokemon TCG Live extract when SVRESTYLE_TCGLIVE points at one (the game's 1024x1024 card
fronts, see tcglive_front), else from pokemontcg.io (733x1024), with TCGdex's 600x825 as a last fallback.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from . import cache

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
    "xyp": ("xybsp", 0),
    "bwp": ("bwbsp", 0),
}
CORNER_REF = "sv03-135"   # its pokemontcg.io scan's alpha gives TCG Live fronts the printed card's rounded corners


def card(card_id):
    data = cache.fetch_json(f"{TCGDEX}/cards/{card_id}", cache.path("data", f"{card_id}.json"))
    if not data.get("id"):
        raise ValueError(f"TCGdex has no card {card_id!r}")
    return data


def set_info(set_id):
    return cache.fetch_json(f"{TCGDEX}/sets/{set_id}", cache.path("data", f"set-{set_id}.json"))


def _split(card_id):
    m = re.fullmatch(r"([a-z]+)0*(\d*)(\.5)?-0*(\w+)", card_id)
    if not m:
        raise ValueError(f"unexpected card id {card_id!r}")
    return m.groups()


def pokemontcg_id(card_id):
    """TCGdex id -> pokemontcg.io image path: swsh6-099 -> swsh6/99, sv03.5-065 -> sv3pt5/65, sma-SV33 -> sma/SV33."""
    prefix, num, half, local = _split(card_id)
    return f"{prefix}{num}{'pt5' if half else ''}/{local}"


def tcglive_id(card_id):
    """TCGdex id -> TCG Live card id: swsh6-099 -> swsh6_99, sv03.5-065 -> sv3-5_65, sma-SV33 -> sm11-5_102,
    swshp-SWSH055 -> swshbsp_55, me01-054 -> me1_54."""
    prefix, num, half, local = _split(card_id)
    set_id, offset = TCGLIVE_SETS.get(f"{prefix}{num}", (f"{prefix}{num}{'-5' if half else ''}", 0))
    digits = re.fullmatch(r"[A-Za-z]*0*(\d+)", local)
    if not digits:
        raise ValueError(f"{card_id}: no card number")
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
    front = tcglive_front(card_id)
    if front is None:
        warnings.warn(f"{card_id}: not in the TCG Live extract ({tcglive_id(card_id)}), using pokemontcg.io")
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
