"""Optional record of what the composers draw, by kind, for building foil masks (foil.py).

    with layers.recording() as drawn:
        compose(...)
    drawn["icon"]   # HxW float32 coverage of every energy icon pasted

typeset.stamp and imaging.put_sprite report their coverage here; the composers name the kind of what they are
drawing with `with layers.kind("icon"):` (innermost wins). Outside recording() nothing is kept.
"""
from contextlib import contextmanager

import numpy as np

W, H = 733, 1024
_drawn = None
_kinds = []


@contextmanager
def recording():
    global _drawn
    outer, _drawn = _drawn, {}
    try:
        yield _drawn
    finally:
        _drawn = outer


@contextmanager
def kind(name):
    _kinds.append(name)
    try:
        yield
    finally:
        _kinds.pop()


def record(alpha, x0=0, y0=0):
    """Add coverage `alpha` (h x w, 0..1) whose top-left pixel is (x0, y0) to the current kind's layer."""
    if _drawn is None:
        return
    name = _kinds[-1] if _kinds else "other"
    layer = _drawn.get(name)
    if layer is None:
        layer = _drawn[name] = np.zeros((H, W), np.float32)
    h, w = alpha.shape
    region = layer[y0:y0 + h, x0:x0 + w]
    np.maximum(region, alpha[:region.shape[0], :region.shape[1]], out=region)
