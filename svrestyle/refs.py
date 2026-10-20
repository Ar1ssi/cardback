"""The real Scarlet & Violet scans the frame and type profiles are built from, and what was measured on them.

Every reference is a regular (not Tera, not full art) Basic ex. Positions are at 733x1024. sha = first 16 hex of
the pokemontcg.io scan the measurements were taken on.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TypeRef:
    card_id: str
    name: str
    sha: str
    name_box_right: int      # right edge of the printed name + ex logo, plus margin
    icon: tuple              # name-bar type icon centre (Hough)


TYPES = {
    "Grass": TypeRef("sv02-027", "Wo-Chien ex", "00cd96d806fdd0ea", 438, (664.5, 57.5)),
    "Fire": TypeRef("sv02-040", "Chi-Yu ex", "fbfe55d9194c46a2", 361, (664.5, 58.5)),
    "Water": TypeRef("sv02-061", "Chien-Pao ex", "dc56c451a1fbee5b", 440, (666.5, 58.5)),
    "Lightning": TypeRef("sv03.5-145", "Zapdos ex", "69a1e23bd08ad4fa", 377, (667.5, 63.5)),
    "Psychic": TypeRef("sv03.5-151", "Mew ex", "28d29c4988d6f0ee", 323, (672.5, 65.5)),
    "Fighting": TypeRef("sv03-120", "Klawf ex", "5a269735d4e91df4", 341, (665.5, 57.5)),
    "Darkness": TypeRef("sv03-135", "Absol ex", "e36b07e01eb1bf80", 346, (666.5, 60.5)),
    "Metal": TypeRef("sv01-143", "Iron Treads ex", "7eb8387e12ce0987", 464, (671.5, 57.5)),
    "Dragon": TypeRef("sv05-123", "Raging Bolt ex", "932d57ec587d110c", 463, (672.5, 63.5)),
    "Colorless": TypeRef("sv03.5-115", "Kangaskhan ex", "0319c0f188314a39", 467, (670.5, 61.5)),
}
ICON_R = 23.0    # name-bar type icon radius, rim included

# Type-independent pieces.
ABSOL = TypeRef("sv03-135", "Absol ex", "e36b07e01eb1bf80", 346, (666.5, 60.5))   # ex logo, layout measurements
ALAKAZAM = ("sv03.5-065", "740ba021ae2516bd")   # window bar, weakness row lines, rule box
HOUNDOUR = ("sv03-131", "f7d14b556ad66942")     # a Basic with a clean border (ex cards' art spills over theirs)
CHIEN_PAO = TYPES["Water"]                      # Ability badge (blue panel keys cleanly against red and silver)
