"""Cards TCGdex and Pokemon TCG Live do not have yet: Mega ex prints revealed before their English release, set up
from scans found online so the editor takes them like any Mega Evolution ex print (modern.py: edited in place).

Each card has its data in TCGdex's shape (CARDS) and one scan, downloaded once and checked against its SHA-256:
- an English print (Mega Lucario Z ex, Flowing Fists 058/080, from the official reveal render): resized to the
  layout's 733x1024 and moved onto it (registered on the weakness row and rule box of a Mega ex print, REF);
- a Japanese print (Mega Garchomp Z ex and Mega Absol Z ex, Aura Seeker M7): Serebii's 868x1212 scans, which sit on
  the layout once resized. Their Japanese words are taken off and set again in English (anglicize.py): what is then
  the card "as printed" is that English print, so every later edit finds its text where the composers set it.

Their holofoil is made here too (custom_foil), since TCG Live has no printing of them: see printings().
"""
import copy
import functools
import re

import cv2
import numpy as np
from PIL import Image

from . import cache

W, H = 733, 1024
VERSION = 1            # bump when the scans' preparation (registration, corners) changes
REF = "me01-086"       # Mega Absol ex (Double Rare): the layout the scans are registered on
REG_BOXES = ((30, 862, 715, 925), (250, 925, 715, 1000))   # x0, y0, x1, y1: weakness row and rule box (every Mega ex)
REG_MIN = 0.75         # px: a smaller measured offset is left alone (the scan's own blur)
REG_MAX = 8.0          # px: a larger one is a failed measurement, refused

SETS = {
    "me07": {"id": "me07", "name": "Flowing Fists", "serie": {"id": "me", "name": "Mega Evolution"},
             "cardCount": {"official": 80}, "abbreviation": {"official": "FLO"}, "releaseDate": "2027-02-05"},
    "mej7": {"id": "mej7", "name": "Aura Seeker (Japanese M7)", "serie": {"id": "me", "name": "Mega Evolution"},
             "cardCount": {"official": 78}, "abbreviation": {"official": "M7"}, "releaseDate": "2026-11-27"},
}

MEGA_RULE = "Mega Evolution ex rule: When your Mega Evolution Pokémon ex is Knocked Out, your opponent takes 3 Prize cards."


def _pokemon(card_id, name, local_id, set_id, hp, type_name, stage, evolves, illustrator, attacks, abilities=(),
             weakness=None, resistance=None, retreat=1, translated=False):
    data = {
        "id": card_id, "localId": local_id, "name": name, "category": "Pokemon", "hp": hp, "types": [type_name],
        "stage": stage, "illustrator": illustrator, "rarity": "Double Rare", "regulationMark": "J",
        "suffix": "EX", "attacks": [dict(a) for a in attacks], "abilities": [dict(a) for a in abilities],
        "weaknesses": [weakness] if weakness else [], "resistances": [resistance] if resistance else [],
        "retreat": retreat, "description": MEGA_RULE,
        "set": {"id": set_id, "name": SETS[set_id]["name"], "cardCount": SETS[set_id]["cardCount"]},
        "image": None,
    }
    if evolves:
        data["evolveFrom"] = evolves
    if translated:
        data["translated"] = True          # not drawn: the English is ours, not an official card's
    return data


# id -> {"data": TCGdex-shaped card, "lang": "en" | "ja", "scan": {url, sha256, page}}
CARDS = {
    "me07-058": {
        "lang": "en",
        "scan": {"url": "https://images.squarespace-cdn.com/content/682a3307f6bd950e4b98cbb9/"
                        "7e1245a6-5c7f-4269-b9f7-d884f52215d3/Pokemon_TCG_058_Mega_Lucario_Z_ex_EN_1_png.webp",
                 "sha256": "28854e6e19083c710a57c9b34b0d5b53d71d66b999f212d78d0c5ebc14e6ae30",
                 "page": "https://pokecottage.com/sets/flowing-fists-card-list"},
        "data": _pokemon(
            "me07-058", "Mega Lucario Z ex", "058", "me07", 330, "Metal", "Stage1", "Riolu", "takuyoa",
            abilities=[{"type": "Ability", "name": "Aura Seeker",
                        "effect": "Once during your turn, if this Pokémon is in the Active Spot, you may use this "
                                  "Ability. Search your deck for a Supporter card, reveal it, and put it into your "
                                  "hand. Then, shuffle your deck."}],
            attacks=[{"cost": ["Metal", "Metal"], "name": "Dancing Fist", "damage": "120+",
                      "effect": "If your opponent's Active Pokémon is a Pokémon ex, this attack does 120 more "
                                "damage."}],
            weakness={"type": "Fire", "value": "×2"}, resistance={"type": "Grass", "value": "-30"}, retreat=2),
    },
    "mej7-030": {
        "lang": "ja",
        "scan": {"url": "https://www.serebii.net/card/auraseeker/30.jpg",
                 "sha256": "5b69bb594ba9819e58ad40b2891e4fc21ca290427a155dd4244a312655c35c9b",
                 "page": "https://www.serebii.net/card/auraseeker/030.shtml"},
        "data": _pokemon(
            "mej7-030", "Mega Absol Z ex", "030", "mej7", 270, "Psychic", "Basic", None, "5ban Graphics",
            attacks=[{"cost": ["Psychic", "Psychic", "Colorless"], "name": "Bloody Fear", "damage": "160",
                      "effect": "During your opponent's next turn, they can't attach Special Energy cards from "
                                "their hand to their Pokémon, and they can't play any Stadium cards from their "
                                "hand."}],
            weakness={"type": "Darkness", "value": "×2"}, resistance={"type": "Fighting", "value": "-30"},
            retreat=1, translated=True),
    },
    "mej7-067": {
        "lang": "ja",
        "scan": {"url": "https://www.serebii.net/card/auraseeker/67.jpg",
                 "sha256": "16d78f838470ea3881e731f7514985bb2716beb51bc715655b5cf1897b99f9a6",
                 "page": "https://www.serebii.net/card/auraseeker/067.shtml"},
        "data": _pokemon(
            "mej7-067", "Mega Garchomp Z ex", "067", "mej7", 360, "Dragon", "Stage2", "Gabite", "aky CG Works",
            attacks=[{"cost": ["Water", "Fighting"], "name": "Drago Avenger", "damage": "30×",
                      "effect": "This attack does 30 damage for each {N} Pokémon in your discard pile."},
                     {"cost": ["Water", "Fighting", "Colorless"], "name": "Sonic Edge", "damage": "240",
                      "effect": "This attack's damage isn't affected by any effects on your opponent's Active "
                                "Pokémon."}],
            retreat=1, translated=True),
    },
}


def is_local(card_id):
    return isinstance(card_id, str) and card_id in CARDS


def is_local_set(set_id):
    return set_id in SETS


def card(card_id):
    """The card's data in TCGdex's shape (a copy: callers edit what they read)."""
    return copy.deepcopy(CARDS[card_id]["data"])


def set_info(set_id):
    return copy.deepcopy(SETS[set_id])


def find(name):
    """TCGdex-style briefs of the local cards whose name holds `name` (any case)."""
    name = (name or "").strip().lower()
    return [{"id": cid, "localId": c["data"]["localId"], "name": c["data"]["name"], "image": None}
            for cid, c in CARDS.items() if name and name in c["data"]["name"].lower()]


def japanese(card_id):
    return CARDS[card_id]["lang"] == "ja"


# ---------------------------------------------------------------- scans
def source(card_id):
    """The scan as downloaded (cached), checked against its SHA-256."""
    info = CARDS[card_id]["scan"]
    ext = re.search(r"\.(\w+)$", info["url"]).group(1)
    dest = cache.path("local", "sources", f"{card_id}.{ext}")
    cache.fetch(info["url"], dest, headers={"Accept": "*/*"})    # without it, Squarespace's CDN sends a JPEG
    got = cache.sha256(dest)
    if got != info["sha256"]:
        dest.unlink()
        raise cache.DownloadError(f"{card_id}: {info['url']} is not the scan this card was set up with "
                                  f"(sha256 {got[:12]}..., expected {info['sha256'][:12]}...)")
    return dest


def _gradient(rgb):
    grey = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grey = cv2.GaussianBlur(grey, (0, 0), 1.2)
    return np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))


def offset(rgb, ref):
    """(dx, dy): how far the print's frame (weakness row, rule box) lies from the reference print's, by phase
    correlation of their edges, averaged over REG_BOXES."""
    a, b = _gradient(ref), _gradient(rgb)
    shifts = []
    for x0, y0, x1, y1 in REG_BOXES:
        (dx, dy), _ = cv2.phaseCorrelate(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
        shifts.append((dx, dy))
    return tuple(float(v) for v in np.mean(shifts, axis=0))


def registered(rgb, ref):
    """rgb (733x1024) moved onto the reference's frame when it lies more than REG_MIN px off."""
    dx, dy = offset(rgb, ref)
    if max(abs(dx), abs(dy)) > REG_MAX:
        raise ValueError(f"the scan's frame lies ({dx:.1f}, {dy:.1f}) px off the Mega ex layout: not registered")
    if max(abs(dx), abs(dy)) < REG_MIN:
        return rgb
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(rgb, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def corners():
    """The printed card's rounded corners (alpha, 0..255) at 733x1024, as tcg gives TCG Live fronts."""
    from . import tcg
    dest = cache.path("web-scans", f"{tcg.CORNER_REF}.png")
    if not dest.exists():
        tcg._web_scan(tcg.CORNER_REF, dest)
    return np.asarray(tcg._open(dest).getchannel("A"), np.float32)


def raw_scan(card_id):
    """The print as found, on the layout: float32 RGBA 733x1024 (cached)."""
    from . import tcg
    from .imaging import load_rgba, save_rgba
    dest = cache.path("local", f"v{VERSION}", f"{card_id}-print.png")
    if dest.exists():
        return load_rgba(dest)
    img = Image.open(source(card_id)).convert("RGB").resize((W, H), Image.LANCZOS)
    rgb = registered(np.asarray(img, np.float32), tcg.scan(REF)[..., :3])
    out = np.dstack([rgb, corners()]).astype(np.float32)
    save_rgba(out, dest)
    return out


def scan(card_id):
    """The card as printed, float32 RGBA 733x1024: the English print, or the Japanese one set in English."""
    if japanese(card_id):
        from . import anglicize
        return anglicize.english_print(card_id)
    return raw_scan(card_id)


# ---------------------------------------------------------------- holofoil
@functools.lru_cache(maxsize=8)
def printings(card_id):
    """{'base': record, 'reverse': None}: a Mega ex Double Rare's printing (SvUltra, as TCG Live prints them),
    with this card's own white plate and etch made from its print (custom_foil). Records look like TCG Live's
    MaterialManifest entries, so foil.py edits them as it edits the game's."""
    from . import custom_foil, foil
    return {"base": {"_local": card_id, "bundle": f"local_{card_id}_v{custom_foil.VERSION}",
                     "manifest": "MaterialManifest", "_s": foil.ME_ULTRA, "_f": "HoloFoil_SvUltra_J",
                     "_w": "local-wp", "_e": "local-etch", "_cc": None},
            "reverse": None}

