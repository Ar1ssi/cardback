"""LaMa inpainting (big-lama TorchScript). Only masked pixels are ever changed."""
import numpy as np

from . import cache

MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
MODEL_SHA = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
_model = None


def model_file():
    dest = cache.path("models", "big-lama.pt")
    cache.fetch(MODEL_URL, dest, timeout=600)
    if cache.sha256(dest) != MODEL_SHA:
        dest.unlink()
        raise cache.DownloadError(f"{MODEL_URL}: checksum mismatch, deleted the download; run again")
    return dest


def _load():
    global _model
    if _model is None:
        import warnings
        try:
            import torch
        except ImportError as e:
            raise SystemExit("torch is missing: pip install torch --index-url https://download.pytorch.org/whl/cpu") \
                from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)   # torch.jit.load deprecation notice
            _model = torch.jit.load(str(model_file()), map_location="cpu").eval()
    return _model


def inpaint(rgba, mask):
    """rgba: HxWx4 (float or uint8), mask: HxW bool. Returns float32 RGBA; alpha and unmasked pixels untouched."""
    import torch
    model = _load()
    rgba = np.asarray(rgba, np.float32)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return rgba.copy()
    h, w = mask.shape
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(rgba[..., :3], ((0, ph), (0, pw), (0, 0)), mode="reflect") / 255
    msk = np.pad(mask, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    img *= 1 - msk[..., None]          # LaMa must not see what it replaces, or it leaves ghosts of it
    with torch.inference_mode():
        out = model(torch.from_numpy(img).permute(2, 0, 1)[None].float(), torch.from_numpy(msk)[None, None])
    fill = (out[0].permute(1, 2, 0).numpy()[:h, :w] * 255).clip(0, 255).round()
    result = rgba.copy()
    result[mask, :3] = fill[mask]
    return result
