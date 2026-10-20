"""svrestyle command line.

  python -m svrestyle prepare --fonts DIR            check dependencies and fonts, download and build every asset
  python -m svrestyle card ID [ID ...] --fonts DIR   render cards (TCGdex ids, e.g. swsh6-99)
      --style hybrid|panel   hybrid (default): art runs behind the text; panel: SV translucent type panel
      --no-attacks           leave out abilities and attacks (names, text, costs, damage)
      --out DIR              where PNGs go (default ./out); --debug also writes the art-cleaning mask overlay
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from . import cache, frame, inpaint, refs, swsh, tcg
from . import typeset as ts
from .imaging import save_rgba


def fonts_dir(arg):
    d = arg or os.environ.get("SVRESTYLE_FONTS")
    if not d:
        raise SystemExit("pass --fonts DIR (or set SVRESTYLE_FONTS): a folder of Gill Sans .otf files")
    ts.configure(d)
    return d


def cmd_prepare(a):
    fonts_dir(a.fonts)
    for mod in ("numpy", "PIL", "cv2", "torch"):
        try:
            __import__(mod)
        except ImportError:
            raise SystemExit(f"missing Python package for {mod}: pip install -r requirements.txt")
    print("cache:", cache.CACHE)
    print("model:", inpaint.model_file())
    frame.build_all()
    print("assets built for:", ", ".join(refs.TYPES))


def cmd_card(a):
    from .compose import Options, compose
    fonts_dir(a.fonts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = Options(style=a.style, attacks=not a.no_attacks)
    suffix = ("hybrid" if a.style == "hybrid" else "panel") + ("" if opts.attacks else "-no-attacks")
    failed = 0
    for card_id in a.ids:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                img, report = compose(card_id, opts)
            c = report["card"]
            stem = f"{card_id}-{(c['prefix'] + ' ' + c['name']).strip().lower().replace(' ', '-')}-ex-{suffix}"
            path = out / f"{stem}.png"
            save_rgba(img, path)
            if a.debug:
                scan = tcg.scan(card_id)
                mask, det = swsh.text_mask(scan)
                save_rgba(swsh.overlay(scan, mask), out / f"{stem}-mask.png")
                (out / f"{stem}-report.json").write_text(json.dumps({**report, "art_detection": det}, default=str,
                                                                   indent=1, ensure_ascii=False))
            notes = [str(w.message) for w in caught if "torch.jit" not in str(w.message)]
            print(f"{card_id}: {path}" + "".join(f"\n  warning: {n}" for n in notes))
        except (SystemExit, ValueError, cache.DownloadError) as e:
            failed += 1
            print(f"{card_id}: FAILED: {e}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="svrestyle", description="Sword & Shield V -> Scarlet & Violet ex restyle")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare", help="check dependencies, download the model, build frame assets")
    pp.add_argument("--fonts")
    pp.set_defaults(func=cmd_prepare)
    pc = sub.add_parser("card", help="render one or more cards")
    pc.add_argument("ids", nargs="+", help="TCGdex card ids, e.g. swsh6-99")
    pc.add_argument("--fonts")
    pc.add_argument("--style", choices=("hybrid", "panel"), default="hybrid")
    pc.add_argument("--no-attacks", action="store_true")
    pc.add_argument("--out", default="out")
    pc.add_argument("--debug", action="store_true", help="also write the art mask overlay and a JSON report")
    pc.set_defaults(func=cmd_card)
    a = p.parse_args(argv)
    a.func(a)
