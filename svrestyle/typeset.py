"""Text rendering in the Scarlet & Violet house style (Gill Sans), supersampled and composited as coverage maps."""
import functools
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    return ImageFont.truetype(f"{_font_dir}/{face}.otf", round(size * SS))


def text_width(text, face, size, xscale=1.0, tracking=0.0):
    return (font(face, size).getlength(text) / SS + tracking * max(0, len(text) - 1)) * xscale


def cap_height(face, size):
    b = font(face, size).getbbox("H")
    return (b[3] - b[1]) / SS


def coverage(items):
    """items: dicts(text, face, size, x, base, xscale=1, tracking=0, anchor='l'|'c'|'r') -> HxW float coverage."""
    canvas = np.zeros((H * SS, W * SS), np.float32)
    for it in items:
        text = it["text"]
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
    canvas[..., :3] = canvas[..., :3] * (1 - cov[..., None]) + np.asarray(fill, np.float32) * cov[..., None]


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


def runs_width(runs, size, tracking):
    chars = sum(len(t) for t, _ in runs)
    return sum(font(f, size).getlength(t) / SS for t, f in runs) + tracking * max(0, chars - 1)


def justify(text, size, full_width, natural_tracking):
    """SV effect text is justified by letter-spacing: full lines spread to full_width, the last line keeps the
    natural tracking. -> [(runs, tracking)]"""
    ws, lines = words(text), []
    while ws:
        n = len(ws)
        while n > 1 and runs_width(runs_of(ws[:n]), size, natural_tracking) > full_width:
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


def line_items(runs, tracking, size, x, base):
    """Coverage items for one line of runs."""
    items, cursor = [], x
    for text, face in runs:
        items.append(dict(text=text, face=face, size=size, x=cursor, base=base, tracking=tracking))
        cursor += font(face, size).getlength(text) / SS + tracking * len(text)
    return items
