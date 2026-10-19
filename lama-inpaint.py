"""Inpaint masked pixels with LaMa (big-lama TorchScript), keep the rest of the card untouched.

Usage: python3 -I lama-inpaint.py <big-lama.pt> <card.png> <mask.png> <out.png>
"""
import sys
import numpy as np
import torch
from PIL import Image

model_path, card_path, mask_path, out_path = sys.argv[1:5]
card = Image.open(card_path).convert("RGBA")
rgba = np.asarray(card)
mask = np.asarray(Image.open(mask_path).convert("L")) > 127

H, W = mask.shape
pad_h, pad_w = (-H) % 8, (-W) % 8
img = np.pad(rgba[:, :, :3], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect").astype(np.float32) / 255
msk = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect").astype(np.float32)
img *= 1 - msk[:, :, None]  # the model must not see the glyphs it is replacing

model = torch.jit.load(model_path, map_location="cpu").eval()
with torch.inference_mode():
    out = model(torch.from_numpy(img).permute(2, 0, 1)[None], torch.from_numpy(msk)[None, None])
fill = (out[0].permute(1, 2, 0).numpy()[:H, :W] * 255).clip(0, 255).round().astype(np.uint8)

result = rgba.copy()
result[mask, :3] = fill[mask]
Image.fromarray(result).save(out_path)
print("inpainted", int(mask.sum()), "px ->", out_path)
