"""Verify the de-hyphenation fix: the 3 hyphenated line breaks should now be
joined without the spurious space, and no block should contain '- ' artifacts
from line breaks."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import extractor

PDF = r"C:\Users\joela\Downloads\Designing Data-Intensive Applications, 2nd Edition-1-55.pdf"

pages = extractor.extract_blocks(PDF)
targets = ["analysis-friendly", "higher-level system", "jack-vanlightly.com"]
found = {t: False for t in targets}
artifacts = []

for page_blocks in pages:
    for b in page_blocks:
        t = b["text"]
        for needle in targets:
            if needle in t:
                found[needle] = True
        # leftover "x- y" artifacts (hyphen + space mid-word) — lowercase both sides
        import re
        for m in re.finditer(r"[a-z]- [a-z]", t):
            artifacts.append((b["page_index"] + 1, t[max(0, m.start()-25):m.end()+25]))

for needle, ok in found.items():
    print(f"{'OK ' if ok else 'FALTA'}  {needle!r}")
print(f"\nArtefactos 'x- y' restantes: {len(artifacts)}")
for pg, ctx in artifacts[:10]:
    print(f"  p{pg}: …{ctx}…")

# Unit checks on _join_lines directly
vocab = {"information", "data"}
cases = [
    (["informa-", "tion is key"], "information is key"),        # syllable, in vocab
    (["analysis-", "friendly schema"], "analysis-friendly schema"),  # compound
    (["TCP-", "IP stack"], "TCP-IP stack"),                      # uppercase next
    (["plain line", "second line"], "plain line second line"),   # no hyphen
    (["Hernan‐", "dez Saenz"], "Hernandez Saenz"),          # U+2010 typographic
    (["corre‐", "sponded to"], "corresponded to"),          # U+2010 not in vocab
    (["Local‐", "First Software"], "Local‐First Software"),  # U+2010 + uppercase → keep
]
print("\nUnit _join_lines:")
for inp, expected in cases:
    got = extractor._join_lines(inp, vocab)
    print(f"  {'OK ' if got == expected else 'FAIL'} {inp} -> {got!r}")
