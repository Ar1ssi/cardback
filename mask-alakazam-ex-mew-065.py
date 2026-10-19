"""Build a text mask for Alakazam ex (151 #065, sv3pt5-65) at 733x1024.

Usage: python3 -I mask-alakazam-ex-mew-065.py <card.png> <out-mask.png> <out-overlay.png>
The boxes are hand-placed for this print's 733x1024 scan (images.pokemontcg.io/sv3pt5/65_hires.png);
another card or scan size needs its own boxes. See HANDOFF.md.
Keeps symbols (type icon, energy costs, weakness/resistance/retreat icons, rarity stars);
removes letters and numbers, plus the regulation-mark and set-code badges (text-only badges).
"""
import sys
import numpy as np
from PIL import Image, ImageFilter

src, out_mask, out_overlay = sys.argv[1:4]
im = Image.open(src).convert("RGBA")
rgb = np.asarray(im)[:, :, :3].astype(np.int16)
R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
L = (0.299 * R + 0.587 * G + 0.114 * B)
H, W = L.shape
mask = np.zeros((H, W), bool)


def region(x0, y0, x1, y1, rule, grow):
    sub = np.zeros((H, W), bool)
    ys, xs = slice(y0, y1), slice(x0, x1)
    sub[ys, xs] = rule(R[ys, xs], G[ys, xs], B[ys, xs], L[ys, xs])
    if grow:
        img = Image.fromarray((sub * 255).astype(np.uint8))
        img = img.filter(ImageFilter.MaxFilter(2 * grow + 1))
        sub = np.asarray(img) > 0
    mask[:] |= sub


def dark(t):
    return lambda r, g, b, l: l < t


def light(t):
    return lambda r, g, b, l: l > t


def outlined(dark_t, light_t):
    # Dark glyph fill wrapped in a white keyline: catch both, then grow to bridge the anti-aliasing.
    return lambda r, g, b, l: (l < dark_t) | (l > light_t)


def not_purple(r, g, b, l):
    # The name bar is pink-purple (G well below R and B); glyphs, white outline and
    # the teal/silver ex logo are neutral or green-leaning.
    return (g > r - 28) | (l < 95)


def whole(r, g, b, l):
    return np.ones(l.shape, bool)


# Name bar
region(19, 37, 92, 54, outlined(150, 222), 2)                  # STAGE (dark fill, white outline)
region(92, 35, 116, 56, outlined(150, 222), 2)                 # 2
region(140, 38, 432, 88, not_purple, 1)                        # Alakazam + ex logo
region(527, 38, 638, 90, dark(120), 2)                         # HP 310
region(138, 96, 318, 115, dark(110), 1)                        # Evolves from Kadabra

# Attacks
region(234, 592, 392, 638, dark(120), 2)                       # Mind Jack
region(628, 596, 698, 630, dark(120), 2)                       # 90+
region(54, 638, 682, 700, dark(120), 2)                        # attack 1 text
region(234, 717, 520, 762, dark(120), 2)                       # Dimensional Hand
region(616, 719, 682, 755, dark(120), 2)                       # 120
region(54, 762, 682, 818, dark(120), 2)                        # attack 2 text

# Weakness / resistance / retreat labels and values
region(38, 886, 113, 908, dark(110), 2)                        # weakness
region(144, 882, 183, 908, dark(110), 2)                       # x2
region(225, 886, 303, 908, dark(110), 2)                       # resistance
region(335, 882, 385, 909, dark(110), 2)                       # -30
region(453, 886, 511, 908, dark(110), 2)                       # retreat

# ex rule box
region(290, 921, 459, 939, whole, 0)                           # "Pokémon ex rule" (white on the black pill)
region(507, 928, 684, 952, dark(100), 2)                       # When your Pokémon ex
region(276, 953, 643, 978, dark(100), 2)                       # is Knocked Out, ...

# Footer
region(40, 938, 188, 958, outlined(85, 165), 2)                # Illus. Mitsuhiro Arita
region(40, 958, 120, 991, whole, 0)                            # G regulation mark + MEW EN badge
region(118, 963, 201, 994, outlined(85, 165), 2)               # 065/165
region(213, 1002, 520, 1015, whole, 0)                         # copyright line

Image.fromarray((mask * 255).astype(np.uint8)).save(out_mask)
ov = np.asarray(im).copy()
ov[mask, 0:3] = [0, 255, 0]
Image.fromarray(ov).save(out_overlay)
print("masked px:", int(mask.sum()))
