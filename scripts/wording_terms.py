"""Counts how the ptcg-sim card corpora print each wording term, to keep svrestyle/wording.py honest.

    python scripts/wording_terms.py PATH/TO/out/pkmn-pokemon-cards.json PATH/TO/out/pkmn-trainer-cards.json

For every term in wording.TERMS: how often it prints capitalised and how often lower-case (whole words).
Then the capitalised two-to-four-word phrases the corpora use 20 times or more, for finding terms the list
lacks (attack and Pokémon names show up here too; the list is hand-pruned of those).
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svrestyle import wording  # noqa: E402


def texts(paths):
    out = []
    for p in paths:
        for row in json.loads(Path(p).read_text(encoding="utf8")):
            t = row.get("text")
            out += t if isinstance(t, list) else [t] if isinstance(t, str) else []
    return out


def main(paths):
    blob = "\n".join(texts(paths))
    print(f"{len(blob.splitlines())} text lines\n")
    print("term: printed capitalised / printed lower-case")
    for term, _ in wording.TERMS:
        pattern = r"(?<![\w\-])" + re.escape(term) + r"(?!\w)"
        upper = len(re.findall(pattern, blob))
        lower = len(re.findall(pattern, blob.lower())) - upper
        print(f"  {term}: {upper} / {max(lower, 0)}")
    word = r"[A-ZÉ][\w’'\-é]*"
    phrases = collections.Counter()
    for m in re.finditer(rf"\b{word}(?:\s+(?:{word}|of|the|or|and|in|a|to|for|on)){{1,3}}", blob):
        phrase = re.sub(r"\s+(of|the|or|and|in|a|to|for|on)$", "", m.group(0))
        if " " in phrase:
            phrases[phrase] += 1
    print("\ncapitalised phrases used 20+ times:")
    for phrase, n in phrases.most_common():
        if n < 20:
            break
        print(f"  {n} {phrase}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
