# Textless card art — handoff (2026-10-19)

Spun out of a session on `Ar1ssi/vibecodepokemontcg2` (the web Pokémon TCG simulator); paths marked
"game repo" below live there, not here.

## Ask and status
- User: "Find alakazam ex 151 (not the full art or rare, just the regular one) and see if you can
  remove the text from it."
- **Done as a one-off image**, delivered in chat. Not wired into the game: the user has not said what
  it is for. The output PNG is not committed; regenerating it takes ~15 s and is byte-for-byte
  deterministic (sha256 `7ae45440035ef5441bc4d934c4a6715cd0e0b149a34ce61a7f1a5089a2c67c3c`).

## The card
- 151 #065 Alakazam ex, Double Rare, illus. Mitsuhiro Arita — TCGdex `sv03.5-065`; game repo corpus
  row `out/pkmn-pokemon-cards.json` {name "Alakazam ex", set "151", number "065"}.
- Excluded on purpose: #188 (Ultra Rare, full art) and #201 (Special Illustration Rare).
- Source scan: `https://images.pokemontcg.io/sv3pt5/65_hires.png`, 733×1024 RGBA, corners transparent,
  sha256 `740ba021ae2516bdbb6c51709e54246ce1fc003a33532d4ed62dd7f780facea1` (byte-identical to
  pkmncards' `sv3-5_en_065_std.png`). TCGdex's `high.png` is only 600×825, so it was not used.

## What was removed, what was kept (a call made for the user)
- Removed: every letter and number — STAGE2, name + ex logo, HP 310, "Evolves from Kadabra", both
  attack names / texts / damage, weakness / resistance / retreat labels and values, all ex-rule-box
  text, illustrator, 065/165, copyright line. Also the "G" regulation mark and the "MEW EN" set badge
  (text-only badges).
- Kept: symbols — type icon, attack energy costs, weakness/resistance/retreat icons, rarity stars.
  Removing those too is a few more boxes in the mask script if the user wants bare frame + art.

## Reproduce
```sh
pip install numpy pillow
pip install torch --index-url https://download.pytorch.org/whl/cpu      # ran with 2.14.1+cpu, Python 3.13
curl -sSLo big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
#   205 803 670 bytes, sha256 7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c
curl -sSo card.png https://images.pokemontcg.io/sv3pt5/65_hires.png
python3 -I mask-alakazam-ex-mew-065.py card.png mask.png overlay.png   # 79139 px
python3 -I lama-inpaint.py big-lama.pt card.png mask.png textless.png
```
Keep downloads and outputs outside the repo (scratch dir) so the model and card images never get committed. `overlay.png` paints the mask green over the card:
check it before inpainting.

## How it works
- `mask-alakazam-ex-mew-065.py`: hand-placed boxes, each with a pixel rule, then grown 0–3 px to
  cover anti-aliasing. Rules: dark glyphs on light panels; "not purple" for the name + ex logo
  (glyphs, keyline and teal logo are neutral/green-leaning, the bar is not); dark-fill-or-white-keyline
  for STAGE2, illustrator and card number; whole box for the black rule pill, the badges and the
  copyright strip.
- `lama-inpaint.py`: LaMa (big-lama TorchScript), pads to a multiple of 8, and copies back **only**
  masked pixels. Verified: alpha and every unmasked pixel are byte-identical to the source.

## Pitfalls found (each cost an iteration)
- Zero the masked pixels before inference. Without it LaMa sees the glyphs and leaves ghosts of them
  (faint "Pokémon ex rule", the MEW badge outline).
- A box that touches a frame bevel makes LaMa extend the bevel into the hole ("teeth" in the black
  pill). The pill box stops at y 939, where every edge is still black.
- OpenCV Telea was tried for the pill and rejected: it smeared the grey bevel upward. LaMa with the
  tighter box keeps the pill's dark-teal gradient.
- Threshold masks miss the light anti-aliased edges of grey text on grey (copyright line) and of white
  keylines (065/165): use whole boxes or grow the mask further.

## Open / next
- Ask the user what the image is for. If it should show in game, the game repo's
  `shared/tcgdex/card-art.mjs` already maps cards to own fronts in `client/src/assets/card-art`;
  wiring it in there is a product decision.
- Other cards need their own boxes. Same-era regular cards share this frame, so many boxes would
  likely carry over, but text extents differ per card; doing many cards needs a text detector instead
  of hand-placed boxes.
- If these scripts move into the game repo: they need torch (pip, not `package.json`), so its harness
  wants a `.agent/DECISIONS.md` line for that dependency and a `.agent/MAP.md` line for the new folder.
