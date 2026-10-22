"""Text rendering in the Scarlet & Violet house style (Gill Sans), supersampled and composited as coverage maps."""
import functools
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import layers
from .imaging import grow

W, H = 733, 1024
SS = 4

BOLD, MEDIUM, MEDIUM_ITALIC, REGULAR = "Gill Sans Bold", "Gill Sans Medium", "Gill Sans Medium Italic", "Gill Sans"
COND_BOLD, BOLD_ITALIC, HEAVY_ITALIC = "GillSans Condensed Bold", "Gill Sans Bold Italic", "Gill Sans Heavy Italic"
REQUIRED_FACES = [BOLD, MEDIUM, MEDIUM_ITALIC, REGULAR, COND_BOLD, BOLD_ITALIC, HEAVY_ITALIC]

_font_dir = None


def configure(font_dir):
    """Point at a directory of Gill Sans .otf files named like 'Gill Sans Bold.otf'. Raises if any face is missing."""
    global _font_dir
    from pathlib import Path
    missing = [f for f in REQUIRED_FACES if not (Path(font_dir) / f"{f}.otf").exists()]
    if missing:
        raise SystemExit(f"{font_dir}: missing Gill Sans faces: {', '.join(m + '.otf' for m in missing)}")
    _font_dir = str(font_dir)
    font.cache_clear()


@functools.lru_cache(maxsize=256)
def font(face, size):
    if _font_dir is None:
        raise RuntimeError("typeset.configure(font_dir) first")
    if face.lower().endswith((".ttf", ".otf")):          # a font file by path (stage.evolves_style)
        return ImageFont.truetype(face, round(size * SS))
    return ImageFont.truetype(f"{_font_dir}/{face}.otf", round(size * SS))


def text_width(text, face, size, xscale=1.0, tracking=0.0):
    return (font(face, size).getlength(text) / SS + tracking * max(0, len(text) - 1)) * xscale


def cap_height(face, size):
    b = font(face, size).getbbox("H")
    return (b[3] - b[1]) / SS


NO_GLYPH = "\U000F0000"     # a private-use code point no font maps: it draws the face's empty box


def _glyph(f, ch):
    mask = f.getmask(ch)
    return mask.size, bytes(mask)


def missing_glyphs(text, face):
    """The characters of text the face has no glyph for (Gill Sans: Greek, arrows): they would print as its box."""
    f = font(face, 20)
    box = _glyph(f, NO_GLYPH)
    return sorted({ch for ch in text if not ch.isspace() and _glyph(f, ch) == box})


def coverage(items):
    """items: dicts(text, face, size, x, base, xscale=1, tracking=0, anchor='l'|'c'|'r', thin=0) -> HxW float
    coverage. thin: px taken off each side of every stroke, for a weight between two faces (1 / SS px steps)."""
    canvas = np.zeros((H * SS, W * SS), np.float32)
    for it in items:
        text = it["text"].replace(GLUE, " ")
        if not text:
            continue
        f = font(it["face"], it["size"])
        asc, desc = f.getmetrics()
        xscale, tracking = it.get("xscale", 1.0), it.get("tracking", 0.0)
        pad = 6 * SS
        adv = [f.getlength(text[:i]) + i * tracking * SS for i in range(len(text) + 1)]
        tile = Image.new("L", (int(adv[-1]) + 2 * pad, asc + desc + 2 * pad), 0)
        d = ImageDraw.Draw(tile)
        for i, ch in enumerate(text):
            d.text((pad + adv[i], pad + asc), ch, font=f, fill=255, anchor="ls")
        if xscale != 1.0:
            tile = tile.resize((max(1, round(tile.width * xscale)), tile.height), Image.LANCZOS)
        thin = round(it.get("thin", 0.0) * SS)
        if thin > 0:
            tile = Image.fromarray(cv2.erode(np.asarray(tile), cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * thin + 1, 2 * thin + 1))))
        width = (adv[-1] - tracking * SS) * xscale
        left = it["x"] * SS - {"l": 0, "c": width / 2, "r": width}[it.get("anchor", "l")] - pad * xscale
        top = it["base"] * SS - pad - asc
        t = np.asarray(tile, np.float32) / 255
        x0, y0 = round(left), round(top)
        tx0, ty0 = max(0, -x0), max(0, -y0)
        x1, y1 = min(W * SS, x0 + t.shape[1]), min(H * SS, y0 + t.shape[0])
        if x1 <= x0 + tx0 or y1 <= y0 + ty0:
            continue
        region = canvas[y0 + ty0:y1, x0 + tx0:x1]
        np.maximum(region, t[ty0:ty0 + region.shape[0], tx0:tx0 + region.shape[1]], out=region)
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)


def stamp(canvas, cov, fill, edge=None):
    """Composite coverage in `fill` (RGB or HxWx3). edge: (rgb, radius, opacity) keyline under the fill."""
    if edge is not None:
        color, radius, opacity = edge
        e = grow(cov, radius) * opacity
        canvas[..., :3] = canvas[..., :3] * (1 - e[..., None]) + np.asarray(color, np.float32) * e[..., None]
        layers.record(e)
    canvas[..., :3] = canvas[..., :3] * (1 - cov[..., None]) + np.asarray(fill, np.float32) * cov[..., None]
    layers.record(cov)


def rounded_box(x0, y0, x1, y1, r):
    img = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(img).rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], r * SS, fill=255)
    return cv2.resize(np.asarray(img, np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def star(cx, cy, outer, inner):
    angles = np.pi / 2 + np.arange(10) * np.pi / 5
    radii = np.where(np.arange(10) % 2 == 0, outer, inner)
    img = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(img).polygon([((cx + r * np.cos(a)) * SS, (cy - r * np.sin(a)) * SS)
                                 for a, r in zip(angles, radii)], fill=255)
    return cv2.resize(np.asarray(img, np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- body text: words, italics, justification
def words(text, roman=MEDIUM, italic=MEDIUM_ITALIC):
    """Split effect text into (word, face); reminder text in parentheses is italic, as printed."""
    out, depth = [], 0
    for word in re.findall(r"\S+", text):
        opens, closes = word.count("("), word.count(")")
        face = italic if depth > 0 or opens > 0 else roman
        out.append((word, face))
        depth = max(0, depth + opens - closes)
    return out


def runs_of(line_words):
    """[(word, face)] -> [(text, face)] with words of the same face merged (spaces kept with the run before)."""
    runs = []
    for i, (word, face) in enumerate(line_words):
        piece = word + (" " if i < len(line_words) - 1 else "")
        if runs and runs[-1][1] == face:
            runs[-1] = (runs[-1][0] + piece, face)
        else:
            runs.append((piece, face))
    return runs


# Energy symbols in effect text: TCGdex writes {G}, {R} ...; the text carries ENERGY_MARK in their place, which is
# laid out as a gap ENERGY_ADVANCE x size wide and filled with the symbol afterwards (line_items' marks).
ENERGY_CODES = {"G": "Grass", "R": "Fire", "W": "Water", "L": "Lightning", "P": "Psychic", "F": "Fighting",
                "D": "Darkness", "M": "Metal", "N": "Dragon", "C": "Colorless"}
ENERGY_MARK = ""
# A space no line breaks at: drawn and measured as a space, but part of the word either side (the prints keep a
# number with the word after it, glue_numbers).
GLUE = ""
ENERGY_ADVANCE = 0.93          # the print leaves a symbol 23 px between spaces in 24.9 px body text (130 SV / ME)


def energy_text(text):
    """'Move a Basic {G} Energy' -> ('Move a Basic \\ue000 Energy', ['Grass'])"""
    types = []

    def mark(m):
        types.append(ENERGY_CODES[m.group(1)])
        return ENERGY_MARK
    return re.sub(r"\{([" + "".join(ENERGY_CODES) + r"])\}", mark, text or ""), types


def length(text, face, size):
    """Width of text in px, energy marks included."""
    return (font(face, size).getlength(text.replace(ENERGY_MARK, "").replace(GLUE, " ")) / SS
            + text.count(ENERGY_MARK) * ENERGY_ADVANCE * size)


def runs_width(runs, size, tracking):
    chars = sum(len(t) for t, _ in runs)
    return sum(length(t, f, size) for t, f in runs) + tracking * max(0, chars - 1)


def justify(text, size, full_width, natural_tracking, fit_tracking=None):
    """SV effect text is justified by letter-spacing: full lines spread to full_width, the last line keeps the
    natural tracking. A word joins a line while the line fits at fit_tracking (default: natural_tracking).
    -> [(runs, tracking)]"""
    fit = natural_tracking if fit_tracking is None else fit_tracking
    ws, lines = words(text), []
    while ws:
        n = len(ws)
        while n > 1 and runs_width(runs_of(ws[:n]), size, fit) > full_width:
            n -= 1
        runs = runs_of(ws[:n])
        ws = ws[n:]
        if ws:
            chars = sum(len(t) for t, _ in runs)
            natural = runs_width(runs, size, 0)
            lines.append((runs, (full_width - natural) / max(1, chars - 1)))
        else:
            lines.append((runs, natural_tracking))
    return lines


LOOSE_LIMIT = 2.0          # px of letter-spacing past which a composed line counts as badly loose


def glue_numbers(text):
    """A number stays on the line of the word after it ('30 more damage', '1 of your'), as the prints break."""
    return re.sub(r"(?<!\S)(\d+) (?=\S)", lambda m: m[1] + GLUE, text)


def compose_paragraph(text, size, full_width, natural_tracking, min_tracking):
    """Line breaks for the paragraph as a whole, the way a paragraph composer sets it: of every way to break it,
    the one whose full lines stay closest to the natural letter-spacing (least sum of squared deviations), no line
    squeezed below min_tracking; the last line keeps the natural spacing when it fits. Trainer text breaks like this
    (it squeezes one line hard rather than leave the next loose). -> [(runs, tracking)] as justify."""
    ws = words(text)
    if not ws:
        return []
    n = len(ws)
    best = [None] * (n + 1)              # best[j]: (cost, start of the last line, its tracking) for words[:j]
    best[0] = (0.0, None, None)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            if best[i] is None:
                continue
            runs = runs_of(ws[i:j])
            chars = sum(len(t) for t, _ in runs)
            natural = runs_width(runs, size, 0)
            needed = (full_width - natural) / max(1, chars - 1)
            if needed < min_tracking and i < j - 1:
                break                                     # longer lines are only tighter
            if j == n and needed >= natural_tracking:
                cost, tracking = 0.0, natural_tracking
            else:
                tracking = needed
                cost = (needed - natural_tracking) ** 2 + 1e3 * max(0.0, needed - LOOSE_LIMIT)
            total = best[i][0] + cost
            if best[j] is None or total < best[j][0]:
                best[j] = (total, i, tracking)
    lines, j = [], n
    while j > 0:
        _, i, tracking = best[j]
        lines.append((runs_of(ws[i:j]), tracking))
        j = i
    return lines[::-1]


def line_items(runs, tracking, size, x, base, marks=None):
    """Coverage items for one line of runs. marks: a list that gets the centre x of each energy mark, in order."""
    items, cursor = [], x
    for text, face in runs:
        for i, part in enumerate(text.split(ENERGY_MARK)):
            if i:
                advance = ENERGY_ADVANCE * size
                if marks is not None:
                    marks.append(cursor + advance / 2)
                cursor += advance + tracking
            if part:
                items.append(dict(text=part, face=face, size=size, x=cursor, base=base, tracking=tracking))
                cursor += font(face, size).getlength(part.replace(GLUE, " ")) / SS + tracking * len(part)
    return items
