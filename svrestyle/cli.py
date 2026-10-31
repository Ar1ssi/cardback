"""svrestyle command line.

  python -m svrestyle prepare --fonts DIR            check dependencies and fonts, download and build every asset
  python -m svrestyle gui [--port 8765] --fonts DIR  editor in the browser: edit attacks, abilities, weakness...
  python -m svrestyle card ID [ID ...] --fonts DIR   render cards (TCGdex ids, e.g. swsh6-99)
      Pokemon V become Scarlet & Violet ex cards; regular Sun & Moon / Sword & Shield Pokemon become Mega
      Evolution regular cards; Trainers before Scarlet & Violet (Items, Supporters, Stadiums, Pokemon Tools)
      become Scarlet & Violet Trainers; Energy before Scarlet & Violet (Basic Energy, Special Energy) becomes Mega
      Evolution Energy (--style, --no-attacks and --debug are for Pokemon only; --foil also works on Energy).
      --style hybrid|panel   ex only. hybrid (default): art runs behind the text; panel: SV translucent panel
      --no-attacks           leave out abilities and attacks (names, text, costs, damage)
      --out DIR              where PNGs go (default ./out); --debug also writes the art-cleaning mask overlay
      --foil                 also write the card's TCG Live holofoil: 1024x1024 front, white plate(s), foil.json
  every command: --tcglive DIR   take scans from a Pokemon TCG Live extract (its 1024x1024 card fronts) instead of
                                 pokemontcg.io; or set SVRESTYLE_TCGLIVE. Cached apart from web scans.
"""
import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

from . import (badges, cache, energy, fairy, foil, frame, inpaint, layers, lifted, megaretype, megarule, meframe,
               oldwindow, refs, swsh, tcg, trainer, trainerframe)
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
    with_fairy = ["Fairy"] if fairy.available() else []
    print("ex assets built for:", ", ".join(list(refs.TYPES) + with_fairy))
    meframe.build_all()
    print("Mega Evolution assets built for:", ", ".join(list(meframe.POOL) + with_fairy))
    megarule.badge()
    print(f"Mega badge (Mega Evolution rule) lifted from {len(megarule.SOURCES)} Mega ex prints")
    if tcg.TCGLIVE:
        megaretype.build_all()
        print("Mega ex type change assets built for:", ", ".join(t for t in megaretype.TYPES
                                                               if t in megaretype.SOURCES or with_fairy))
    if not with_fairy:
        print("Fairy needs the TCG Live extract (its Energy coins): not built")
    trainerframe.build_all()
    print("Energy frames are built when first used (Team Rocket's Energy, Ignition Energy, Mega Evolution Basic Energy)")
    for layout in oldwindow.LAYOUTS:
        oldwindow.reference(layout)
    for badge, (era, _) in badges.POOL.items():
        badges.look(badge, trainer.ERA_ART[era])
    print("Trainer frames built for:", ", ".join(trainerframe.KINDS), "; older layouts:", ", ".join(oldwindow.LAYOUTS),
          "; badges:", ", ".join(badges.POOL))
    kinds = [k for k in lifted.SOURCES if lifted.load(k) is not None]
    print("print glyphs lifted for:", ", ".join(kinds) or "none (set in Gill Sans)")


def is_v(card_id):
    return tcg.card(card_id).get("name", "").endswith(" V")


def is_trainer(card_id):
    return tcg.card(card_id).get("category") == "Trainer"


def file_slug(name):
    """'Type: Null' -> 'type-null': a colon would name an NTFS stream, not a file."""
    return re.sub(r"[^\w-]+", "-", name.strip().lower()).strip("-")


def is_energy(card_id):
    return tcg.card(card_id).get("category") == "Energy"


def render_energy(card_id, out, want_foil=False):
    """-> (PNG path, foil.json path or None)."""
    with layers.recording() as drawn:
        img, report = energy.compose_card(card_id)
    card = report["card"]
    stem = f"{card_id}-{file_slug(card['name'])}-energy"
    path = out / f"{stem}.png"
    save_rgba(img, path)
    foil_path = None
    if want_foil:
        built = energy.build_foil(card, img, drawn)
        if built["base"] or built["reverse"]:
            foil_path = foil.save(built, out, stem, card_id)
    return path, foil_path


def render_trainer(card_id, out, want_foil=False):
    """-> (PNG path, foil.json path or None)."""
    with layers.recording() as drawn:
        img, report = trainer.compose_card(card_id)
    card = report["card"]
    stem = f"{card_id}-{file_slug(card['name'])}-trainer"
    path = out / f"{stem}.png"
    save_rgba(img, path)
    foil_path = None
    if want_foil:
        if card.get("ace"):
            built = foil.build_ace(card_id, card, img, drawn, trainer.read_card(card_id))
        else:
            built = foil.build_trainer(card_id, card, img, drawn)
        if built["base"] or built["reverse"]:
            foil_path = foil.save(built, out, stem, card_id)
    return path, foil_path


def cmd_card(a):
    from . import mecompose
    from .compose import Options, compose
    fonts_dir(a.fonts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = Options(style=a.style, attacks=not a.no_attacks)
    suffix = ("hybrid" if a.style == "hybrid" else "panel") + ("" if opts.attacks else "-no-attacks")
    failed = 0
    for card_id in a.ids:
        try:
            if is_energy(card_id):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    path, foil_path = render_energy(card_id, out, a.foil)
                notes = [str(w.message) for w in caught if "torch.jit" not in str(w.message)]
                print(f"{card_id}: {path}" + (f"\n{card_id}: foil {foil_path}" if foil_path else "")
                      + "".join(f"\n  warning: {n}" for n in notes))
                continue
            if is_trainer(card_id):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    path, foil_path = render_trainer(card_id, out, a.foil)
                notes = [str(w.message) for w in caught if "torch.jit" not in str(w.message)]
                print(f"{card_id}: {path}" + (f"\n{card_id}: foil {foil_path}" if foil_path else "")
                      + "".join(f"\n  warning: {n}" for n in notes))
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                v = is_v(card_id)
                with layers.recording() as drawn:
                    img, report = compose(card_id, opts) if v else mecompose.compose_card(card_id, opts.attacks)
            c = report["card"]
            slug = file_slug(c["prefix"] + " " + c["name"])
            stem = f"{card_id}-{slug}-ex-{suffix}" if v else f"{card_id}-{slug}-me" + ("" if opts.attacks else "-no-attacks")
            path = out / f"{stem}.png"
            save_rgba(img, path)
            if a.foil:
                built = foil.build(card_id, "ex" if v else "me", img, drawn, c)
                print(f"{card_id}: foil {foil.save(built, out, stem, card_id)}")
            if a.debug and v:
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


def cmd_gui(a):
    from . import gui
    fonts_dir(a.fonts)
    if a.ptcg_sim:
        os.environ["SVRESTYLE_PTCG_SIM"] = a.ptcg_sim
    gui.serve(a.port, open_browser=not a.no_browser)


def main(argv=None):
    p = argparse.ArgumentParser(prog="svrestyle", description="Restyle older cards: V -> Scarlet & Violet ex, "
                                                           "regular -> Mega Evolution, Trainers -> Scarlet & Violet, "
                                                           "Energy -> Mega Evolution")
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
    pc.add_argument("--foil", action="store_true", help="also write TCG Live holofoil textures (needs --tcglive): "
                                                         "1024x1024 front, white plates, <card>-foil.json")
    pc.set_defaults(func=cmd_card)
    pg = sub.add_parser("gui", help="open the editor in your browser (served on 127.0.0.1)")
    pg.add_argument("--fonts")
    pg.add_argument("--port", type=int, default=8765)
    pg.add_argument("--no-browser", action="store_true", help="just print the address")
    pg.add_argument("--ptcg-sim", metavar="DIR", help="the ptcg-sim checkout \"Send to ptcg-sim\" writes custom sets "
                                                      "into (or set SVRESTYLE_PTCG_SIM)")
    pg.set_defaults(func=cmd_gui)
    for sp in (pp, pc, pg):
        sp.add_argument("--tcglive", metavar="DIR", help="take card scans from this Pokemon TCG Live extract "
                                                         "(1024x1024 fronts; or set SVRESTYLE_TCGLIVE)")
    a = p.parse_args(argv)
    if a.tcglive:
        tcg.TCGLIVE, cache.SOURCE = a.tcglive, "tcglive"
    a.func(a)
