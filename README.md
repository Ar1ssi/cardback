# textless-card-art

Tools for re-making Pokemon TCG card scans.

## svrestyle: older cards -> Scarlet & Violet / Mega Evolution look

Give it TCGdex card ids; it picks the layout from the card:
- **Pokemon V** (Sword & Shield Basic V) -> **Scarlet & Violet ex**: the old frame and text are removed and the art
  rebuilt (LaMa inpainting), then the SV frame for the card's type, Gill Sans lettering, ex logo, rule box and footer
  are set from the card's real data.
- **Regular Pokemon** (Sun & Moon, Sword & Shield; Basic, Stage 1, Stage 2) -> **Mega Evolution regular card**: the art
  window moves over as is (it sits in the same place in all three eras), a Stage card's old evolution portrait is
  inpainted out and its pre-evolution picture moved into the Mega Evolution portrait ring, and the card is set on a
  blank Mega Evolution frame of its type with name, HP, Pokedex line, abilities/attacks, weakness row, flavour text
  and footer.
- **Trainers** (Item, Supporter, Stadium, Pokemon Tool; Sword & Shield back to Base Set) -> **Scarlet & Violet
  Trainer**: the old art moves into the Gen 9 window and the card is set on the Gen 9 frame of its kind (see
  Trainers below).
- **Energy** (Basic and Special, Sun & Moon to Black & White frames) -> **Mega Evolution Energy**: a Basic Energy
  becomes the Mega Evolution Basic Energy of its type with its own set code and number in the footer; a Special Energy
  keeps its art in the window of a Mega Evolution Special Energy, with its text reworded as the new cards say it (see
  Energy below).

```sh
pip install -r requirements.txt            # torch: --index-url https://download.pytorch.org/whl/cpu
export SVRESTYLE_FONTS=/path/to/gill-sans  # licensed fonts, not in this repo
export SVRESTYLE_TCGLIVE=/path/to/TCGLive_Extract   # optional: scans from Pokemon TCG Live (see below)
python -m svrestyle prepare                # once: model + reference scans + frame assets (~/.cache/svrestyle)
python -m svrestyle card swsh6-99 --no-attacks          # Galarian Slowking V -> ex, full art, no attacks
python -m svrestyle card swsh9-040 swsh1-138 --style panel
python -m svrestyle card swsh3-88 sm10-19 sma-SV33      # Tyranitar, Kartana, Shiny Vault Kartana -> Mega Evolution
python -m svrestyle card base1-91 ex3-87 dp1-114 swsh1-156   # Bill, Mr. Briney's Compassion, Speed Stadium, Air Balloon
python -m unittest discover tests
```

### Scans from Pokemon TCG Live

By default scans come from pokemontcg.io. With `--tcglive DIR` (any command) or `SVRESTYLE_TCGLIVE=DIR`, they are
taken from a Pokemon TCG Live extract instead: the game's 1024x1024 card fronts, decoded from
`DIR/cardart/bundles/<set>_en_<NNN>` (needs `pip install UnityPy`) or read from `DIR/WebApp/assets/cards/<set>_<N>.webp`.
A front is the 733x1024 card at the same scale plus 145 px of bleed on each side, so it is cropped, given the printed
card's rounded corners and used with the same layout constants; the reference cards the frames are built from come
from it too. TCGdex ids map to TCG Live ids by rule (`swsh6-99` -> `swsh6_99`, `sv03.5-065` -> `sv3-5_65`) plus
`tcg.TCGLIVE_SETS` (`sma-SV33` -> `sm11-5_102`, `swshp-SWSH055` -> `swshbsp_55`, `mep-013` -> `mebsp_13`). A card the extract lacks falls back
to pokemontcg.io with a warning. Everything built from these scans is cached under `<cache>/tcglive/`.

### Holofoil (`--foil`)

With a TCG Live extract, `card ... --foil` also writes the card's holofoil the way TCG Live draws it: a 1024x1024
`<card>-front.png` (the card plus bleed, the game's texture layout), white plates `<card>-wp.png` (base printing)
and `<card>-wp-ph.png` (reverse holo), and `<card>-foil.json` naming the shader, material and pattern for each
(`foil.py`). The foil is the new era's: ex cards get Scarlet & Violet ex foil (SunPillar, TEX_SouthernCross); Mega
Evolution cards get SvHolo on the art window if the old card's base printing was foil, and every one gets the Gen 8
reverse holo (below). Plates start from the game's own: the old card's plate for the art (the Pokemon stays out of the
foil), Absol ex's or me1 Abra's / Meganium's for the frame; the composers record every stamp and sprite by kind
(`layers.py`), so the HP, icons, badges and footer drawn on the card are cut out of the plate exactly.

Every card's foil starts from its own plate in the extract. `foil.field` takes what is printed off it (text, icons,
figure): the same set's printings of the same kind with the same foil shapes - a Sword & Shield Darkness reverse
holo's symbols on chevrons, a Sun & Moon Grass reverse's leaves - show the foil under this card's text, and each is
fitted to this card's own tone around every gap. What no card shows (the art window) is bridged smoothly. A
restyled reverse holo is not: it is the Gen 8 one.

The Gen 8 (Sword & Shield) reverse holo is the standard on every 1-prize Pokemon and every Trainer but ACE SPEC
(`foil.gen8_plate`, `foil.gen8_trainer_plate`): FlatSilver on TCG Live's chevron pattern with the card's type's
symbol in each tile (Trainers: Poké Balls), taken from a Basic of each type whose plate prints it with the fewest
marks (`GEN8_REFS`; Ultra Ball for Trainers), its cut-outs filled from the pattern one tile (120 px) up or down. Fairy,
which Sword & Shield does not print, is Psychic's pattern with the Fairy symbol set in each tile (`foil.gen8_swapped`).
The pattern shows where the card's own reverse plate shows foil: a restyled card's Mega Evolution plate (me1 Abra's
or Meganium's, this card's icons, badge and footer cut out), a Scarlet & Violet or Mega Evolution print's own plate,
a Trainer's Gen 9 plate of its kind (sv01-168 Item, sv01-166 Supporter, sv01-167 Stadium, sv01-169 Tool: its header
lettering and rule pill stay, the chevrons fill its pattern field). Every such card gets it, whether the old print
had a reverse holo or not (a print with no foil at all takes its era's frame from a reference print: Sprigatito /
Floragato, Abra / Meganium). What stays: a holo (a card whose base printing is foil keeps it; the set uses the holo),
a print's own reverse that is not the generic FlatSilver one (`foil.generic_reverse`), ACE SPEC Trainers' AceFoil
(`build_ace`), ex cards and Energy. A push turns holofoil on for every card that takes it (`gui.standard_foil`).

### Scarlet & Violet / Mega Evolution cards: edited in place

In the editor, a Scarlet & Violet or Mega Evolution Pokemon (regular or ex) keeps its own card: only what you edit is
taken off and set again (`modern.py`). Name/HP and the weakness row are found by setting the printed values again
with the composers (they match these prints within a few pixels) and taking what stands out there, each printed
shape whole; abilities and attacks by what stands out from the panel in the text zone. LaMa rebuilds what was taken
off, except the weakness row, which is blended along its pixel rows (LaMa copies the labels into it). On a plain
panel, a blot LaMa makes up in a hole is covered with the panel beside it. New blocks follow the Mega Evolution
layout (regular cards; Scarlet & Violet regulars share it) or the Absol ex layout (ex), in the ink the old block
was printed in, read from its lines of glyphs (dark or white, and its keyline when outlined over full art);
an edited ex name keeps the card's own ex logo. With holofoil on, the card's own TCG Live printings are re-made: the
same shaders and the card's own plates, edited the way each plate prints that kind of ink: ink the plate shows (a
cut-out name, an ex name's lit letters, the Stage tab's letters) is taken off and set again at the printed value;
ink the plate runs through (most reverse holos' names and "Evolves from") leaves it as printed. In abilities,
attacks and the weakness row, the old cut-outs (energy circles, the badge) take the card's own foil field, so a
dropped cost leaves no circle behind, and new icons are cut like the printed ones; what every printing cuts (the
row's lines) stays cut. The etch is
carried over, and the front keeps the game's real bleed. In effect text, {G} {R} {W} {L} {P} {F} {D} {M} {N} {C} print as energy symbols (any
layout).

### Fairy

Neither Scarlet & Violet nor Mega Evolution prints Fairy, so its look is made from Psychic's (`fairy.py`; needs the TCG
Live extract). TCG Live gives every type a default Energy coin, one design in each type's colours with the symbol
artwork the cards' icons print (`cosmetics/.../CN_TCGL_default_<type>-d`):
- colours: Psychic's turned into Fairy's the way the Psychic coin's turn into the Fairy coin's pixel for pixel (in
  Oklab, a lightness shift, a chroma gain and a hue turn, each fitted per step of lightness). The coin holds the
  glossy disc's mid and dark tones: past them (a frame's pastel panel) only the hue turns, or the panel goes hot pink
  and its mottling washes out. Greys (silver, white, the icons' black) stay; a colour turned out of sRGB loses chroma;
- the symbol: the Fairy coin's, solid (the coin shades it in 3-D), at the scale the Psychic coin's eye lands on the
  Psychic icons' (an icon of radius r prints the coin's art at r / 164; the same on Grass's); in effect text the bare
  symbol sits where the lifted Psychic, Grass and Darkness symbols do (15.1 px art radius at full size);
- icons: the Psychic Mega Evolution icon with its eye matched (the Psychic coin's eye, fitted to the print) and taken
  off with the light ring every icon prints round its symbol, filled from the disc round it, recoloured (keeping the
  disc's own light and shade: the coin's flatter disc would dull its gloss), and the Fairy symbol set in its place in
  the icon's black, ringed with the light the Psychic print has round its eye (read off it: about +0.13 lightness over
  2.5 px on the name-bar icon);
- frames: Psychic's recoloured with Fairy's icon: the Mega Evolution regular frame (a Sun & Moon or XY-era layout
  Fairy card restyled, a Scarlet & Violet / Mega Evolution card retyped), the Scarlet & Violet ex bar and panel, the
  Mega ex frame (name bar, panel tint, glitter);
- Basic Energy: the Mega Evolution Psychic Energy (`mee-005`) recoloured, its four symbols (the ball's, the name line's
  bare one, the icons top right and in the footer) made Fairy's; its reverse plate is Psychic's (the ball prints as
  a plain disc there);
- Fairy as a weakness, resistance, cost, `{Y}` in text, and a Special Energy's icons, everywhere the other types go.
  "Follow the type" gives Fairy Sun & Moon's weakness and resistance: Metal, and Darkness at the new era's -30.
The simulator's custom-set schema (`shared/custom-sets/custom-set-schema.mjs`, `ENERGY_TYPES`) must list Fairy before
a Fairy card goes into a set: a type it does not take makes it reject the whole set (and a published set vanishes from
the site). Send to ptcg-sim, Publish and Push all refuse such cards, before rendering anything, while the simulator
checkout's schema leaves the type out; deploy the schema change before publishing.

### Stage: Basic, Stage 1, Stage 2, Mega

Every card in the editor (restyled ex, restyled Mega Evolution, edited in place) can be made Basic, Stage 1 or
Stage 2 (`stage.py`). A Stage card gets the modern pieces top left: the STAGE tab, the silver ring with the
pre-evolution's portrait, and the "Evolves from" strip (regular cards also the swoosh in the name bar); they sit at the
same place on every SV / ME card, regular and ex. The portrait is lifted whole from a SV / ME card TCGdex lists as
evolving from that Pokemon (they all print the same picture on the same disc); name a card id under "Portrait from
card" to choose: a SV / ME Stage card gives its portrait, any other card the middle of its art. Mega is a custom stage: the
Stage 1 pieces (ring, portrait, "Evolves from") with the tab reading MEGA, set in Verdana Bold widened and slanted
forward to match the printed BASIC / STAGE lettering (`verdanab.ttf`; without it Tahoma Bold, then Gill Sans Bold),
letter by letter with the printed tabs' 2 px gaps and centred where they centre theirs, on the card's own tab with its
lettering taken off. When no SV / ME card evolves from the Pokemon (none does from a Stage 2, so a Mega Charizard ex
made to evolve from Charizard has none), the portrait is the Pokemon's Pokemon HOME render (PokeAPI): the prints'
portraits are those renders in the same poses, so it is set as they are (its canvas 0.9 of the disc wide, the figure
centred) and inked like them (a dark line round the figure and along its sharpest shading, colours a little richer
and deeper). After it come, in order: the XY "M ... EX" card of the Pokemon (the figure keyed off its gold disc), the
Pokemon's official artwork (PokeAPI), and the middle of one of its cards' art. A card made Basic
gets the BASIC tab and the pieces taken off (LaMa; the plain left border is blended top to bottom). A Stage card
keeps its own strip (Mega ex cards end it under their "Mega-Evolved form of" badge): only the old line is blended
away along the strip's rows. The line is set like the print, which is not Gill Sans: Segoe UI Bold Italic narrowed
and tightened matches it glyph for glyph (`segoeuiz.ttf`, looked up in the fonts folder and the system font folders;
without it, Gill Sans Bold Italic). Only the tab's
lettering is swapped, so each card keeps its own tab and border. On the holofoil plates the ring and strip take the
SV ex frame's silver value on ex cards and are left out elsewhere; the portrait, lettering and "Evolves from" line
are cut out.

### Mega Evolution rule

Tick "Mega Evolution rule" in the editor (edit field `mega_rule`) and any Pokemon card - a Mega ex print, a card made
Mega, a restyled card - carries the rule every XY "M ... EX" card printed in a box of its own: "When 1 of your Pokémon
becomes a Mega Evolution Pokémon, your turn ends." It is set the way Gen 9 sets a rule about how a card plays
(`megarule.py`). Gen 9 keeps prize rules (ex, Mega ex) in the box at the bottom right and hangs Tera's rule under the
name bar, over the art: a pearl panel with an iridescent rim under the "Evolves from" strip, labelled by a tab on the
bar above it (Charizard ex sv03-125; Teal Mask Ogerpon ex sv06-025 shows the same place on a Basic). Mega Evolution
prints its Mega mark at that place, the "The Mega-Evolved form of ..." badge, so the Mega rule is that panel with the
badge as its label: the badge sits on the panel's top-right corner, the rule's first line beside it and its second
under it, the way the bottom rule box sets its first line beside its pill.

- Panel: fitted on Tera's: hung from the strip (y 117.5-166), left end rounded and clear of the portrait ring (the
  ring stays on top of it), right end leaning like the strip's and tucked under the badge's lobe, as long as its two
  lines (the rule is shorter than Tera's, which fills the width). Text Gill Sans Bold at Tera's size and line places.
- Colours: its rim takes the badge lettering's colours (pink, violet, blue, green) laid along its own length, its
  pearl body a tenth of the badge's band; both are read off the card's own badge, so a Mega Hyper Rare's is gold.
- Badge: a Mega ex print (every rarity and promo prints it) keeps its own, on top of the panel. Any other card gets
  one rebuilt from 16 Mega ex Double Rares of me01 and me02: their median (`prepare` builds it), its lettering taken off
  the band (the band's colour along x times its shading down y), "The Mega-Evolved form of", which every print sets
  alike, put back as printed, and the Pokemon's name set in Gill Sans Heavy Italic, white with an outline in the
  lettering's colours: what a card made Mega evolves from, else its name without "Mega". A card that is not a Mega
  gets "Mega Evolution rule" on the badge instead.

Nothing under it is taken off or moved: abilities and attacks keep their places (a print edited in place keeps its
own). The holofoil plates leave the panel and a rebuilt badge out; a printed badge keeps its foil. The simulator does
not read the rule: a card exported with it plays as it did.

### Type

A regular Pokemon (one Prize: no ex, V or other rule), a Pokemon V restyled as ex, a Mega Evolution ex and a Scarlet &
Violet ex (not a Tera ex) can change type in the editor. A card restyled as a Mega Evolution card or as ex is simply set
on the new type's frame; a Mega ex or Scarlet & Violet ex is recoloured on its own print (below). A V's art keeps the
V frame's type gradient behind its text, so a V made another type has that gradient moved to the new type's colour
(`swsh.retint`): the frame lays its type's colour over
the art from none at y 721 to all of it at the weakness row's band (y 876), straight in between, measured on the 226
regular V prints in the TCG Live extract (each type's colour is its prints' at the bottom; Fairy's is Psychic's in
Fairy's colours).

Attack costs, weakness and resistance stay as they are when the type changes. "Follow the type" (beside the type) sets
them as the new type's cards print them (`follow.py`): the card's own type's Energy in its costs and effect text becomes
the new type's, and it takes the weakness and resistance most of that type's cards print (the Mega ex Double Rares' for
a Mega ex, the Mega Evolution regular cards' for the rest), its weakness keeping the card's own value. Dragon pays with
other types' Energy: a card made Dragon keeps its costs and prints no weakness, and a Dragon card made another type
pays that type's for every coloured Energy. Each click starts from the card as printed, so changing type again maps
the printed costs, not the last ones.

Energy and damage alone changing (costs, `{X}` in effect text, an attack's damage, whatever the reason) keep the
print of a Scarlet & Violet or Mega Evolution card (`inplace.py`): each printed cost icon of a type that changes is
found in the cost column and covered with the new type's at the same place and size (the old one filled from round it
first: its rim reaches half a pixel past the sprite's, and would ring the new one); each printed symbol in the text is
found by our symbol's shape on the print's ink, filled, and set again in the block's own ink; every attack's printed
damage is found by its figures in the attacks' order down the card (a "30" also matches the end of a "130" below
it), and a changed one rebuilt by LaMa round it and set again at its right edge and baseline. Setting the block again
would wipe a full art's picture under the words. A print on which not all of it is found is set again as before. An
outlined print's keyline (full art) is as wide as the print's own, measured ring by ring round its glyphs.

**Mega ex** (`megaretype.py`; needs the TCG Live extract: the white plates tell the panel, the glitter and what lies
over them apart). Measured on every Mega ex Double Rare of me01 to me05 and the promos printed on their frame:
- the name bar is opaque, one gradient per type, the same on each of its cards: the bar under the card's own print is
  moved from the card's own bar (fitted smooth from its print-free pixels) to the new type's (fitted on the type's
  cards whose bars agree); a pixel's share of bar against the frame's silver pieces is measured per layout and, along
  the pieces' edges, per card. The name and ex logo are yellow on every type; the HP dark, white on Darkness;
- the text panel is the type's colour multiplied over the faded art, the same top colour on each of a type's cards: a
  change there is a ratio per channel, the words re-inked (white on Darkness, dark elsewhere). Darkness prints no red
  under its panel, so a Darkness card's art red is read from its green and blue (a table learnt on every other type's
  panels; about 13 levels off the print, and no better from the card's own art above the panel, which shows other
  colours than under it);
- the glitter frame (the strip left of the panel, the weakness row's band, the footer, a sliver by the portrait) is
  the type's own glitter texture: pixels in the old type's glitter colours take the new type's; the row's icons keep
  theirs;
- Ultra Rare and Special Illustration Rare frames (white lettering over the art) and the Mega Hyper Rare's gold carry
  no type colour: their name-bar icon changes, and a Mega Hyper Rare's etch, which carries the icon's symbol.

**Scarlet & Violet ex** (the same module; Double Rares of sv01 to sv10.5 and the non-Mega ex of me02 to me05): the
same name bar (its name and HP dark, white on Darkness; the ex logo the same on every type) and the same panel. Below
the panel, round the weakness row, the rule box and the footer, and in the strips beside it, is no glitter but a band:
the type's colour, lighter or darker from place to place, with grain, and the card's own art reaching over it. It is not
the same pixels on each of a type's cards (the sets render it up to some 20 levels apart, and art runs over it
anywhere), so a pixel the old type's colour explains there (the median of its cards where they show band, smooth;
within its grain) moves by the change of colour, its grain scaled down to the new type's; on Darkness's dark band any
darker shade is band. The frame's pieces on the band (the rules, the rule box and its pill, the footer's boxes, and
their dark outline) are found where every type's cards agree and kept, as are the row's icons, art unlike the band, and
the footer's print: dark with a white keyline on every type (on Darkness told by the keyline, its letters being the
band's colour). The weakness row's words are re-inked when Darkness is involved. Its Ultra Rare, Special Illustration
Rare, Hyper Rare and Shiny Ultra Rare change their icon alone; Tera ex (their rule in a silver box under the name bar)
are left out.

The holofoil follows: the white plate cuts re-inked white words out (to Darkness) or closes them (from it). Every
other edit then starts from the recoloured print, which is cached (`modern.cleaned` keys its LaMa cleanup on the print
itself). The etch of an ex whose abilities, attacks or weakness row are set again follows them too: the old icons'
etched rings and the Ability badge's pill go, the new ones get the card's own (a badge, Mega Dragonite ex's).

**Regular Scarlet & Violet / Mega Evolution cards** keep their own print too
(`retype.py`): every regular card of a type prints the same frame (`meframe.template`), so what the card prints that
its type's frame does not (words, numbers, icons, the footer's boxes, the Stage pieces, a Trainer's medallion) is
found against its own background and set on the new type's frame. Words take the new type's ink, each pixel at its
share of ink (Darkness prints white, the others dark); everything else keeps its colours, its rim blended onto the new
ground by its contrast. Energy icons are found from the card's data (its costs matched in the cost column, the
weakness row's icons where the composers set them); the Ability badge and red Ability names by their red, the names
in a white keyline on a dark panel and none on a light one, as the Mega Evolution layout sets them. Silver parts
(border, Pokedex strip, the row's rules, the Stage tab) stay as printed. Attack costs, weakness and resistance are
left as they are ("Follow the type" changes them). With holofoil on, the reverse holo follows the ink:
TCG Live's plates cut white lettering out (Darkness) and run the foil through dark lettering (every other type),
so a card going to or from Darkness has its name, HP, text and the row's labels cut out, or filled with the foil
around them (`foil.lettering_cutouts`); energy symbols, the badge, the rules and the footer stay as printed.

Fit to the print: a Scarlet & Violet print sits up to a pixel off the Mega Evolution templates, so it is moved onto
its template first (matched on the borders and the row's rules) and back after, and each Stage piece is matched where
the print has it (a Scarlet & Violet swoosh sits a few pixels right). The background under the print is the old
template, shaded like the print round it. Energy icons are discs fitted to the print's own rim, covered whole (a
Colorless icon's white on a light panel too) with a geometric anti-aliased edge; the footer boxes are found on the
print and their edges unmixed against the box's dark outline or white ring, the white kept neutral.

The Mega Evolution frame templates take the weakness row's lower rule and the panel's bottom rim from the median of
the type's prints (every card's print reaches over them, and the fills blurred them away), the Pokedex strip and its
shadow from the prints too (each pixel from the prints with no dex text there), and fill the weakness row's value
boxes with the row's own grain from further along it; the rebuilt templates are cached as `me-v4`.

### Energy

Old Energy becomes Mega Evolution Energy (`energy.py`; its frames are cut from TCG Live scans, so `--tcglive` is the way to run it).
- **Basic Energy** (Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Fairy): the type's own Mega Evolution
  Basic Energy (`mee-001`..`008`), whose art is the type's. Only the footer (set code, number) is set again; there is
  nothing to edit but the frame. Fairy's is Psychic's made Fairy's (see Fairy below); Dragon and Colorless have no
  Mega Evolution Basic Energy and are refused.
- **Gold Basic Energy** (the editor's Frame: "Gold Hyper Rare"): the type's gold Hyper Rare Basic Energy instead
  (Paldea Evolved's Grass `sv02-278` and Water `sv02-279`; Lightning `sv01-257`, Fighting `sv01-258`, Fire `sv03-230`,
  Psychic `sv03.5-207`, Darkness `sv06.5-098`, Metal `sv06.5-099`), its footer set again the same way with the print's
  three gold stars moved after the number, and the card marked Hyper Rare. Fairy's is Psychic's gold print made
  Fairy's (`fairy.gold_energy`): its art window and the header's swoosh in Fairy's colours, the gold as printed, the
  title line's symbol, the ball's and both icons Fairy's (the same places as on the Mega Evolution Basic Energy). Its
  foil is the gold print's own (SvUltra, plate and etch, `energy._gold_foil`): the plate leaves the new set box out and
  foils the stars where they now stand; the etch's lines run on under the old footer print (taken from the other gold
  prints, whose etches share them there, and one line period along the row where all print) and round the title
  symbol, and the new box, number, stars and Fairy's symbol are etched as the print etches its own.
- **Special Energy**: the old art goes into the window of Team Rocket's Energy (`me02.5-217`: art, text panel), or, for
  a full-art print (Secret Rare and up), under the text as on Ignition Energy (`me02-124`). The art window is found on
  the old scan from its frame's edges (Sun & Moon, Sword & Shield, XY and Black & White frames; others are refused
  with the size found); a full-art print's old text is painted out of its usual place. The text is reworded ("Double
  Colorless Energy provides {C}{C} Energy." -> "As long as this card is attached to a Pokémon, it provides {C}{C}
  Energy."), set tighter than the Trainers' (29 px lines, 50 px between paragraphs), and the Energy it provides are the
  name bar's icons (up to three; set them in the editor when the text names none, e.g. "every type"). Name and icons
  sit where the Mega Evolution Special Energy prints set them (the footer's icons at the attack cost icons' size).
- **Typed Special Energy** (the editor's Frame: "Mega Evolution typed"): a Special Energy that provides one type takes
  that type's Mega Evolution typed Special Energy whole (Telepathic Psychic Energy `me03-088`, Growing Grass Energy
  `me03-086` and the other six: `energy.TYPED_TEMPLATES`): its art (the type's symbol on a glass sphere), Rare grade
  and foil (SvHolo on the swirl round the sphere), with the card's name, text and footer. The type before "Energy" in
  the name prints as its symbol, as on the prints ("Charming Fairy Energy" -> "Charming {Y} Energy"; `energy.typed_name`,
  the symbol lifted off the print's own name). Fairy's is Telepathic Psychic Energy made Fairy's (`fairy.typed_energy`):
  the eye painted out of the sphere (LaMa), Fairy's symbol inked in the eye's own gloss under the print's glare, then
  Fairy's colours; built once and cached. Made for the Fairy one: Wonder Energy (`xy5-144`) named Charming Fairy Energy.
- **Foil** (`energy.build_foil`): a full-art Special Energy keeps its old print's foil on the art and takes Ignition
  Energy's round it; a regular one gets SvHolo on the art when the old print was foil and the template's reverse holo
  (header lettering, text panel) when it had one; a Basic Energy the template's reverse holo when the old print had one.
  TCG Live has no foil for some old Basic Energy: the card is set without, with a warning.
- **Refused**: Scarlet & Violet and Mega Evolution Energy (already in this look), TCG Pocket cards, and an Energy whose
  TCGdex record has no text. A custom set takes Basic and Special Energy.

### Trainers

Older Items, Supporters, Stadiums and Pokemon Tools become Scarlet & Violet / Mega Evolution Trainers
(`trainer.py`). Technical Machines are Tools, as they are from Sword & Shield on; a card TCGdex types Item but whose
text is a Tool's (attaching itself, or lending an attack: Curse Powder, the e-Card Cubes) is a Tool too. The first sets
print no type: the kind comes from the rule the text holds, else Item (with a warning).

- **Frame**: every SV / ME Trainer of a kind prints the same frame, so `trainerframe.py` builds each kind's from the
  median of real prints (name, text and footer taken off; the rule pill and tabs kept). `prepare` builds them.
- **Art**: Sun & Moon to HeartGold SoulSilver cards have one art window per era (`trainer.ERA_ART`). Older scans are
  photographs that sit a few px off and a few percent larger or smaller, so `oldwindow.py` registers each card's frame
  to its layout's mean frame (Base Set, e-Card, EX, Diamond & Pearl, and XY Evolutions' redrawn Base Set layout) and
  carries the layout's window over. The art is scaled to cover the Gen 9 window. Era marks over the art (Single /
  Rapid / Fusion Strike, TAG TEAM, Team Plasma) are found by their learned masks and painted out with LaMa
  (`badges.py`).
- **Text** from TCGdex: the old rule sentences the Gen 9 pill prints instead are taken out (a Supporter whose text
  needs its place beside the Active Pokemon keeps it); spelled-out types print as energy symbols; a choice's options
  print as hanging bullets; straight quotes become curly. The effect is justified by letter-spacing and broken for the
  paragraph as a whole, its first baseline on the prints' grid. Fossils and dolls get their HP in the bar and "a N-HP
  Basic {C} Pokemon" in the text; Technical Machines their attacks, set like the Mega Evolution cards'.
- **ACE SPEC** (Scarlet & Violet, 30 Trainers): edited in place, `svrestyle/ace.py`. Their pink frame with the ACE SPEC
  banners differs set to set, so no template: the card keeps its own scan, and only the name, text and illustrator can
  change (not the kind). What changed is found as white print in its zone, rebuilt by LaMa and set again in white; the
  text sits where the print has it. Their holofoil (AceFoil, `foil.build_ace`) keeps the card's own plate: the old print's cut-outs are filled from the card's foil field and the new text is cut at the value the plate gives it. The ex and V logos in text print as plain "ex" and "V". Older ACE SPEC cards (Black &
  White, XY) are still refused.
- **Refused**: full-art, gold and other special prints (no art window to move), older ACE SPEC and Prism Star cards, GX-attack
  Z-Crystals and Team Flare Hyper Gear (rules no Gen 9 frame prints), Chaos Tower (played either way up), Classic
  Collection reprints, TCG Pocket cards, and sets of no known layout.

Gill Sans has no Greek letters: a name like "Miracle Sphere α" renders with a warning and an empty box for the letter;
edit the text in the editor.

### Editor (GUI)

```sh
python -m svrestyle gui            # opens http://127.0.0.1:8765/ in your browser (--port, --no-browser)
```

On Windows, `start-site.bat` starts the editor (in its own "svrestyle server" window, unless something already
serves the port) and shares it through a Cloudflare quick tunnel (needs `cloudflared`). The
`https://….trycloudflare.com` address prints in its window and is written to `SVRESTYLE_CACHE\tunnel.log`; it
changes every time the tunnel starts.

Load a card by its TCGdex id, or import a ptcg-sim custom-set export to get every older card in it listed. "Search cards…" finds cards on TCGdex by name and filters (card type, Pokemon type, stage, rule box, Trainer type,
regulation mark, rarity, set, HP range) and adds the ones you pick to the list, no ids needed; it offers Pokemon and
Trainer and Energy cards the editor takes (no TCG Pocket or promo sets whose ids it cannot read). For each
card you can change the name (and regional prefix), HP, type (regular Pokemon, restyled ex, Mega ex, Scarlet & Violet ex; "Follow the type" sets the costs, weakness and resistance that go with it), stage, "Evolves from" and portrait, illustrator and flavour
text, add, remove and
reorder abilities and attacks, pick each attack's energy cost from the type icons, set its damage (a number with an
optional +, × or −) and text, and set weakness (type and ×2 / +10…+40), resistance (type and −10…−40) and
retreat (click + / −, or a retreat symbol to remove it). With `--tcglive DIR`, tick Holofoil to re-make the card's
TCG Live foil with every render: the plates follow the edits (costs, Ability badge, weakness, resistance and retreat
icons are cut out where they are drawn); preview the foil on the card or the white plate itself, per printing (holo /
reverse holo), and download the card with its foil textures as a zip (the custom-set zip includes them too). The preview re-renders as you
edit, in exactly the card layout the command line produces; download one PNG, or every card of an imported set as a
zip (your edits included). The server only listens on 127.0.0.1 and needs no packages beyond requirements.txt.
A Trainer has its own fields: name, a Supporter's character, kind (Item, Supporter, Stadium or Tool), HP (fossils and
dolls), illustrator and effect text (a paragraph per line), and its Abilities or attacks; holofoil works too (below). A Scarlet & Violet or Mega Evolution Trainer is edited too: it keeps its art and is set again, text and
all, on its kind's Gen 9 frame (an inline ex logo in its text prints as plain "ex").
A Special Energy has its name, illustrator, text, the Energy it provides (the name bar's icons) and its frame
(art and text panel, or full art); a Basic Energy only its frame (Mega Evolution, or gold Hyper Rare). Holofoil
works on Energy too.

The foil preview is TCG Live's own: its TPCi/Cards3D shaders (the extract's WebApp translation to WebGL,
`WebApp/web/data/foil/`) and MAT_Cards3D_* materials, with their spectrum, shine, glitter and pattern textures
from the extract's shaders bundle (`pokemon_Pokemon TCG Live/shadersbundle/Texture2D/`, lossless), drawing the
render's front, plates, etch and cold foil (`svrestyle/static/foil.js`, the same port the ptcg-sim client runs).
The card turns with the pointer for the light, else sways slowly. three.js 0.185.0 is vendored in
`svrestyle/static/three/` (MIT, LICENSE.txt there); the page needs WebGL2.

Changes: every render lists what differs from the printed card (the restyle itself, name, HP, type, stage, "Evolves
from", portrait, each ability and attack by name: cost, damage, text, added, removed, reordered; weakness,
resistance, retreat, illustrator, flavour text); the card list counts them per card. View the card edited, as
printed, or both side by side. "Changelog (.md)" writes every card's changes as Markdown, and each zip download
holds the same CHANGELOG.md.

Projects: every card you load joins the card list; the Cards panel adds several by id at once (commas or spaces;
those TCGdex does not know, or that are not Pokémon, stay in the box with the reason) and × takes one out with its
edits (Undo puts it back); the find box above the list narrows it to the cards whose name or id holds what you type
(Enter opens the first). Drag a row to move it, or right click it to send it to the top or bottom; the list's
order is the order Render all, the zip and the change log use. The page saves itself 0.3 s after each change - the list, every
card's edits and options, the card open and the holofoil switch - and opens it again on its own next time. Work
with no name yet is saved as "<set> <date time>" (or "Untitled <date time>"); type a name and Save to rename it. New starts an empty project (the open one is saved first and stays in the list);
it saves itself with its first change, or under the name typed and Save. Saved projects (`SVRESTYLE_CACHE/projects/*.json`) can be opened
or deleted from the same bar, from any browser that reaches the server. History lists the project's earlier versions
and the deleted projects, each with Restore. A version is kept every 10 minutes while you work, when work starts again
after a break, and before each Save over a project or Restore, so a Restore can be undone the same way; the newest 100
of each project stay (`projects/.history/<project>/`). Delete moves a project to `projects/.trash/` (the newest 100
stay); restoring one puts it back under its name, or "<name> (restored)" if another project has that name now.
All projects (.zip) merges every saved project into one list (a card in several projects comes from the most
recently saved one, edits included; Energy is left out) and downloads it as the usual zip of renders. The server builds
it in the background and the page shows the progress (a first render takes 10-20 s a card; tunnels cut a long request),
so the 60-card limit of the other downloads does not apply; a card that cannot be rendered is listed in
`NOT-INCLUDED.txt` and the rest still come. While the page follows the build the button reads Cancel: the build stops
after the card it is rendering and nothing is downloaded. Reloading the page and clicking again joins a build still
running. Finished zips
wait in `SVRESTYLE_CACHE/exports/` (the last 3).

Wording help: effect text (abilities, attacks, Trainer text) gets the TCG's fixed terms in their printed capitals as
you type - "active pokemon" becomes "Active Pokémon", "knocked out" "Knocked Out", "energy card" "Energy card" - and
the server does the same to every saved edit (`svrestyle/wording.py`; the list is the terms the ptcg-sim card corpora
print capitalised, counted by `scripts/wording_terms.py`). Names and flavour text are left as typed. "Suggest wording"
under each effect field asks Claude (model `claude-haiku-5-5`) for the same effect in printed-card wording; the
suggestion shows with its notes and nothing changes until you click Use. With `ANTHROPIC_API_KEY` in the server's
environment it calls the Anthropic Messages API (the key never reaches the page). Without a key it runs the server
machine's Claude Code CLI headless (`claude -p`, on that machine's Claude Code login): no tools, no MCP servers, and
`--safe-mode`, so the machine's CLAUDE.md, hooks and plugins stay out; at most 2 at once and 20 a minute, since the
tunnel makes the site public. `SVRESTYLE_CLAUDE` names the `claude` executable if it is not on `PATH`. With neither,
the button is disabled.

Working together: everyone with the same project open edits one copy. Each change goes to the server as a patch of
only the cards and fields that changed, and the others' pages take it in about half a second (long polls, which
pass through Cloudflare quick tunnels; event streams do not). Two people editing different fields, or different
cards, both keep their work; the same field at once, the later one wins. Text being typed is never overwritten
while it is unsent. The header shows how many people have the project open.

### Custom sets for ptcg-sim

The list is also a custom set the simulator (Ar1ssi/vibecodepokemontcg2) plays. Type a set code (2-5 capitals or
digits, the code a decklist prints: `4 Tyranitar HBW 1`) above the list and click "Download for ptcg-sim (.zip)". The
zip holds one folder, `custom<code>/` (`customhbw/`): `set.json` with every card in TCGdex's record shape (name, HP,
type, stage, attacks, Abilities, weakness, resistance, retreat, Trainer text; your edits applied), each card as
`<nnn>/high.webp` (733x1024) and `low.webp`, numbered in list order, and the CHANGELOG. `set.json` always lists the cards in the standard set order, whatever order they were added in (numbers and ids stay put): Pokemon by type A-Z, then evolution line A-Z (a line sits together, Basic, Stage 1, Stage 2; it takes its lowest card's type, `LINE_TYPES` overrides one, and a Basic nothing evolves from keeps its own type), then Trainers (Item, Supporter, Stadium, Tool, each A-Z; ACE SPEC last), then Energy (`export.display_order`). Unzip it into the simulator's
`client/src/assets/custom-sets/` (the running simulator picks it up, no restart): the set shows under the deck builder's Custom pill, its cards
import from a decklist by that code, and the rules engine reads their text like any printed card's (the simulator's
TCGdex proxy serves the folder as TCGdex records). Each card in `set.json` also keeps the TCGdex card it was made from
with your edits (`custom`), so "Import custom set…" opens the `set.json` again with everything restored. A Mega card's
stage is the slot it evolves into (one above its pre-evolution, looked up on TCGdex; Stage 1 when that fails, noted in
the CHANGELOG). A card with Holofoil on (needs `--tcglive`) also brings its TCG Live foil: `foil.json` and
`foil/<nnn>-front.webp` with the white plates, the files the simulator's `import-foil.py` makes, so the simulator's
holofoil shows your edits (its foil layer draws the card from that front texture, not from `high.webp`).
`svrestyle/export.py` is the code.

With the simulator on the same machine, skip the zip: set `SVRESTYLE_PTCG_SIM` to its checkout (or start the GUI with
`--ptcg-sim DIR`; `start-site.bat` sets it when the simulator is in `Downloads\vibecodepokemontcg2-main`) and "Send to
ptcg-sim" appears under the set code. It writes the same files straight into the simulator's
`client/src/assets/custom-sets/<set id>/`; the simulator watches that folder, reloads the set while it runs and
refreshes open deck builders. Only cards whose edits or options changed since the last send are rendered (Shift+click
renders all of them: do that once after updating the editor, whose new rendering a kept card would not show; a set
exported before records kept image digests has its images taken as they are, and the next send adds them; a card's
holofoil is kept only when it is the one the record names, so a foil put in the set another way, with `import-foil.py`,
is made again by the first send that meets it, about 4 s a card), only changed files are written, card folders past the
end of the list are removed, and so are the `foil.json` rows and `foil/` textures no card uses any more. "Keep ptcg-sim in sync" sends after
every render of a card in the list. Games already running keep the cards they loaded. The button shows, and the route
answers, only in a browser on the machine running the editor: visitors through the Cloudflare tunnel cannot write
into the simulator, unless the server has a publish key. Set `SVRESTYLE_PUBLISH_TOKEN` to a random string of 16 or
more characters (`python -c "import secrets; print(secrets.token_urlsafe(24))"`, then `setx SVRESTYLE_PUBLISH_TOKEN
<it>` and restart the site); a tunnel visitor then sees a "Publish key" field, and once the key is typed there the
browser keeps it and sends it with each request (`X-Svrestyle-Key`), so Send to ptcg-sim and Publish to site work
from that browser too. Give the key only to people you would let change the hosted site.

The hosted simulator (Render) cannot see that folder, so "Publish to site" sends the set as above and then runs the
simulator's `scripts/custom-sets/publish.mjs`, which uploads it to the site's R2 bucket with rclone (only files the
bucket lacks go up); the site checks the bucket every 10 s and refreshes open deck builders, no deploy needed. The
button shows when `CUSTOM_SETS_R2_REMOTE` names the bucket's rclone remote (`start-site.bat` sets
`r2:ptcg-foil/custom-sets` when rclone is installed), Node.js is on PATH, and the simulator checkout has the script.
Credentials stay in rclone's config. To take a set off the site: `node scripts/custom-sets/publish.mjs --unpublish
<set id>` in the simulator.

Send and Publish run as jobs on the server that the page follows (a first render takes 10-20 s a card, an upload minutes,
and a tunnel cuts any one request at about 100 s): the status line shows the card being made, then the files going out,
then the upload, and Cancel stops the job. Nothing changes if you cancel before the first file is written; cancelling a
publish stops the upload and the site keeps the copy it had (ptcg-sim already has the new one). A page reloaded
meanwhile gets the job back by clicking the same button again.

A send or publish never takes a card out of a set that is already there: the cards the list does not name stay in the
set (the set often holds more than the open project, since it is every project merged), each with its number, edits
and files as it was. Deleting a card from the list does not delete it from the set; remove its folder in the
simulator's `custom-sets` folder (and its entry in `set.json`) for that. Only a set whose cards this editor did not
write (they name no card they were made from, so a push cannot send them again) and a site copy longer than the list
are refused until you say so: "HB has 299 cards in ptcg-sim; this list has 88: 211 would be removed (Pinsir ex, ...)".
"Keep ptcg-sim in sync" never asks: it stops and says why, and the Send button does the replacing.

A card keeps its number (its id, `customhb-123`) whatever its place in the list: the list's order is only how it looks
in the editor. `set.json` lists the cards in the order the simulator should show them (`export.display_order`):
Pokemon (by type: Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Colorless; within a type
Basic, Stage 1, Stage 2, each A-Z), then Trainers (Item, Supporter, Stadium, Tool, each A-Z, the ACE SPEC Trainers last,
A-Z), then Energy by number (ACE SPEC Energy among it). The simulator's set browser shows custom sets in that order.

"Push all" (top right) pushes every saved project, merged like "All projects (.zip)", as one set. It asks for the set
code, works out what would change without rendering anything and lists it: the cards that are new, the ones changed
since they were pushed (with what changed: HP, an attack's damage, its options), and the ones no saved project has,
which the push removes. Cards already in the set keep their numbers (a deck names a card by its number), the new ones
go on the end; taking cards out of the middle renumbers the ones after them, and the popup says how many. "Push to
ptcg-sim" sends exactly the list shown, rendering only the cards listed (the list is built with the test the send
makes, so it cannot promise a card is unchanged that the send will render); Shift+click renders every card again.
"Push and publish to site" then uploads it.

Options: `--style hybrid` (default, art behind the text) or `panel` (SV translucent type panel) - ex only;
`--no-attacks`; `--debug` (ex: mask overlay + JSON report); `--out DIR`.

Supported types: Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Fairy, Dragon, Colorless (Fairy,
which neither target era prints, needs the TCG Live extract: see Fairy below). Tested on the ex side with swsh6-99, swsh9-040, swsh1-138, swsh4-43, swsh3-19,
swsh7-110, swsh8-113, swsh1-1, swsh1-141, swsh10-078, swsh3-99, swsh12.5-065, swshp-SWSH055, swsh10-101, swsh6-58,
swsh6-80, swsh6-97; on the Mega Evolution side with 22 Sun & Moon / Sword & Shield cards (all types but Lightning),
and checked against real me1 cards re-set by the same code (within 1-4 px).

How it works, module by module:
- `swsh.py` - masks the V frame regions; finds the text block by its white glyphs with a black keyline, the
  Ability badge by colour, and cost icons by circle detection on their fixed slots; LaMa rebuilds the art.
- `frame.py` - builds the SV frame from real scans (`refs.py`): border, window bar, weakness row and rule box
  (type-independent), and per type the name bar (name/HP removed), energy icon, panel palette and ink colour.
- `compose.py` / `typeset.py` - lays out name, HP, abilities, attacks (justified effect text, italic reminder
  text, damage), weakness row and footer, with constants measured on SV scans.
- `lifted.py` - HP, damage and weakness-row values, and the energy symbols in effect text, in the print's own
  glyphs: SV / ME figures are proportional and unlike any Gill Sans face, and the symbols print bare in the text's
  ink, so each glyph is lifted off scans (cut out, aligned, averaged at 4x) and placed as fitted on them (figures by
  their advances; symbols from their gap, found by matching the words either side to our layout of the card's own
  text). Built once by `prepare` or the first render (about a minute from the TCG Live extract); a value with a glyph
  never printed (+20) falls back to Gill Sans, a symbol to the icon's symbol in a disc.
- Energy icons (attack costs, weakness, resistance, retreat, and the ex name-bar type icon) are the Mega Evolution
  ones on both layouts, lifted from real me1 scans (`meframe.COST_ICONS`, `meframe.ICON`); on the Mega Evolution
  layout they also take Mega Evolution's sizes and positions.
- `meframe.py` - blank Mega Evolution frames: Mega Evolution cards of one type share the same bar, panel and footer
  texture, so each printed pixel of a base card is taken from another card of the type that has nothing printed
  there (LaMa only for what every card covers); plus the silver Stage 1 / Stage 2 pieces from Kadabra and Alakazam.
- `megaretype.py` - a Mega ex's or Scarlet & Violet ex's print recoloured on another type (see Type above); `inplace.py` - Energy that
  changes type swapped where the print has it; `follow.py` - the costs, weakness and resistance that go with a type.
- `fairy.py` - Fairy's colours, symbol, icons and frames, made from Psychic's (see Fairy above).
- `retype.py` - a Scarlet & Violet / Mega Evolution regular card's print lifted off its type's frame and set on
  another's (see Type above).
- `stage.py` - Stage pieces, evolution portrait (found on TCGdex) and "Evolves from" line, for every layout.
- `megarule.py` - the Mega Evolution rule as Gen 9 sets such a rule: the panel under the "Evolves from" strip and
  the Mega badge (lifted from the Mega ex prints) that labels it.
- `edit.py` / `gui.py` / `gui.html` - the editor: which fields can change and the checks on them, the local HTTP
  server, and the page.
- `mecompose.py` - the Mega Evolution layout: art window, portrait, Pokedex line (PokeAPI), the vertical spacing of
  abilities and attacks fitted on 45 me1 cards, flavour text; text drawing is shared with `compose.py`.

For Claude Code sessions on this repo the `sv-restyle` skill (`.claude/skills/sv-restyle/SKILL.md`) runs this.

## Older one-off scripts
`mask-alakazam-ex-mew-065.py` + `lama-inpaint.py`: text removal for Alakazam ex (151 #065); see `HANDOFF.md`.
