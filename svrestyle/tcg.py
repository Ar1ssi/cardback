"""Card data (TCGdex) and card scans (pokemontcg.io 733x1024 first, TCGdex 600x825 as a fallback)."""
import re
import warnings

import numpy as np
from PIL import Image

from . import cache

TCGDEX = "https://api.tcgdex.net/v2/en"
W, H = 733, 1024   # every layout constant in this package is measured at this size


def card(card_id):
    data = cache.fetch_json(f"{TCGDEX}/cards/{card_id}", cache.path("data", f"{card_id}.json"))
    if not data.get("id"):
        raise ValueError(f"TCGdex has no card {card_id!r}")
    return data


def set_info(set_id):
    return cache.fetch_json(f"{TCGDEX}/sets/{set_id}", cache.path("data", f"set-{set_id}.json"))


def pokemontcg_id(card_id):
    """TCGdex id -> pokemontcg.io image path: swsh6-099 -> swsh6/99, sv03.5-065 -> sv3pt5/65, sma-SV33 -> sma/SV33."""
    m = re.fullmatch(r"([a-z]+)0*(\d*)(\.5)?-0*(\w+)", card_id)
    if not m:
        raise ValueError(f"unexpected card id {card_id!r}")
    prefix, num, half, local = m.groups()
    return f"{prefix}{num}{'pt5' if half else ''}/{local}"


def scan(card_id, expect_sha=None):
    """The card scan as float32 RGBA at 733x1024."""
    dest = cache.path("scans", f"{card_id}.png")
    if not dest.exists():
        try:
            cache.fetch(f"https://images.pokemontcg.io/{pokemontcg_id(card_id)}_hires.png", dest)
        except cache.DownloadError:
            base = card(card_id).get("image")
            if not base:
                raise
            warnings.warn(f"{card_id}: no 733x1024 scan, using TCGdex's 600x825 (softer result)")
            cache.fetch(f"{base}/high.png", dest)
    if expect_sha and not cache.sha256(dest).startswith(expect_sha):
        warnings.warn(f"{card_id}: scan differs from the one the layout was measured on; check the output")
    img = Image.open(dest).convert("RGBA")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img).astype(np.float32)
