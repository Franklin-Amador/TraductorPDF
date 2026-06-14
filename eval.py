"""Harness de evaluación A/B para el traductor.

Uso:
    python eval.py run --label baseline [--pages 1-55] [--pdf RUTA]
    python eval.py diff baseline despues-fix
    python eval.py scan baseline

`run`  traduce el PDF de referencia y guarda las traducciones por bloque en
       eval_runs/<label>.json (no genera PDF: compara a nivel de texto).
`diff` compara dos runs bloque a bloque (clave: página + bbox) y muestra
       solo los bloques cuya traducción cambió.
`scan` lista bloques sospechosos de un run: glyph ⁇ residual, palabras
       duplicadas adyacentes, ratios de longitud anómalos, trigramas en loop.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
from collections import Counter

import extractor
import translator

DEFAULT_PDF = (
    r"C:\Users\joela\Downloads"
    r"\Designing Data-Intensive Applications, 2nd Edition-1-55.pdf"
)
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_runs")

_DUP_WORD_RE = re.compile(r'\b(\w{3,})\s+\1\b', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pages(spec: str, n_pages: int) -> set[int]:
    """ "1-10,15" -> {0..9, 14} (0-based). Sin spec: todas."""
    if not spec:
        return set(range(n_pages))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a) - 1, int(b)))
        else:
            pages.add(int(part) - 1)
    return {p for p in pages if 0 <= p < n_pages}


def _block_key(block: dict) -> str:
    x0, y0, x1, y1 = (round(v, 1) for v in block["bbox"])
    return f"p{block['page_index']}:{x0},{y0},{x1},{y1}"


def _run_path(label: str) -> str:
    return os.path.join(RUNS_DIR, f"{label}.json")


def _load_run(label: str) -> dict:
    path = _run_path(label)
    if not os.path.exists(path):
        sys.exit(f"No existe el run '{label}' ({path}). Corre: python eval.py run --label {label}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    pdf = args.pdf or DEFAULT_PDF
    if not os.path.exists(pdf):
        sys.exit(f"PDF no encontrado: {pdf}")

    print(f"Extrayendo bloques de: {os.path.basename(pdf)}")
    all_pages = extractor.extract_blocks(pdf)
    wanted = _parse_pages(args.pages, len(all_pages))

    blocks = [b for p in sorted(wanted) for b in all_pages[p]]
    texts = [b["text"] for b in blocks]
    print(f"Páginas: {len(wanted)}/{len(all_pages)}  ·  Bloques: {len(blocks)}")

    print(f"Cargando modelo {args.src}->{args.tgt}…")
    src_sp, tgt_sp, ct2 = translator.load_model(args.src, args.tgt)

    t0 = time.time()
    last_pct = -1

    def progress(done, total):
        nonlocal last_pct
        pct = done * 100 // total
        if pct != last_pct:
            last_pct = pct
            print(f"\r  Traduciendo… {pct}% ({done}/{total})", end="", flush=True)

    outputs = translator.translate_batch(texts, src_sp, tgt_sp, ct2, progress)
    duration = time.time() - t0
    print(f"\n  Listo en {duration:.1f}s")

    entries = []
    for block, src, out in zip(blocks, texts, outputs):
        entries.append({
            "key": _block_key(block),
            "page": block["page_index"] + 1,
            "src": src,
            "out": out,
        })

    os.makedirs(RUNS_DIR, exist_ok=True)
    data = {
        "meta": {
            "label": args.label,
            "pdf": pdf,
            "pages": args.pages or "all",
            "src": args.src,
            "tgt": args.tgt,
            "duration_s": round(duration, 1),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_blocks": len(entries),
        },
        "blocks": entries,
    }
    path = _run_path(args.label)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"Guardado: {path}")

    _scan_entries(entries)


# ---------------------------------------------------------------------------
# scan — heurísticas de bloques sospechosos
# ---------------------------------------------------------------------------

# TOC dot-leaders (". . . . ." or "......") — stripped before analysis so they
# don't trigger the repeated-trigram detector or skew length ratios.
_DOT_LEADER_RE = re.compile(r'(?:\.\s*){4,}|\s{8,}')


def _suspicions(src: str, out: str) -> list[str]:
    src = _DOT_LEADER_RE.sub(' ', src).strip()
    out = _DOT_LEADER_RE.sub(' ', out).strip()

    flags = []
    if "⁇" in out:
        flags.append("glyph-unk")
    if _DUP_WORD_RE.search(out):
        flags.append("dup-word")

    words = out.split()
    if len(words) >= 6:
        trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if trigrams and trigrams.most_common(1)[0][1] >= 2:
            flags.append("loop-trigram")

    ls, lo = len(src), len(out)
    if ls > 20:
        ratio = lo / ls if ls else 0
        if ratio > 1.8:
            flags.append(f"largo×{ratio:.1f}")
        elif ratio < 0.45:
            flags.append(f"corto×{ratio:.1f}")
    return flags


def _scan_entries(entries: list[dict], show: int = 40) -> None:
    passthrough = sum(1 for e in entries if e["out"].strip() == e["src"].strip())
    suspects = []
    for e in entries:
        flags = _suspicions(e["src"], e["out"])
        if flags:
            suspects.append((e, flags))

    print(f"\n— Resumen: {len(entries)} bloques · {passthrough} passthrough "
          f"· {len(suspects)} sospechosos —")
    for e, flags in suspects[:show]:
        print(f"\n[p{e['page']}] {', '.join(flags)}")
        print(f"  SRC: {_short(e['src'])}")
        print(f"  OUT: {_short(e['out'])}")
    if len(suspects) > show:
        print(f"\n… y {len(suspects) - show} sospechosos más")


def cmd_scan(args) -> None:
    data = _load_run(args.label)
    print(f"Run '{args.label}' · {data['meta']['date']} · {data['meta']['n_blocks']} bloques")
    _scan_entries(data["blocks"], show=args.show)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def cmd_diff(args) -> None:
    a = _load_run(args.label_a)
    b = _load_run(args.label_b)
    map_a = {e["key"]: e for e in a["blocks"]}
    map_b = {e["key"]: e for e in b["blocks"]}

    only_a = [k for k in map_a if k not in map_b]
    only_b = [k for k in map_b if k not in map_a]
    changed = []
    for k, ea in map_a.items():
        eb = map_b.get(k)
        if eb and ea["out"].strip() != eb["out"].strip():
            changed.append((ea, eb))

    print(f"Comparando '{args.label_a}' ({a['meta']['date']}) vs "
          f"'{args.label_b}' ({b['meta']['date']})")
    print(f"  Bloques comunes con cambios: {len(changed)}")
    print(f"  Solo en {args.label_a}: {len(only_a)} · Solo en {args.label_b}: {len(only_b)}")

    for ea, eb in changed:
        src_a, src_b = ea["src"], eb["src"]
        print(f"\n[p{ea['page']}]")
        if src_a.strip() != src_b.strip():
            print(f"  SRC A: {_short(src_a)}")
            print(f"  SRC B: {_short(src_b)}")
        else:
            print(f"  SRC  : {_short(src_a)}")
        print(f"  A: {_short(ea['out'], 110)}")
        print(f"  B: {_short(eb['out'], 110)}")

    if only_a or only_b:
        print("\n(Bloques sin pareja — la estructura de extracción cambió; "
              "revisa si es esperado.)")
        for k in (only_a[:5] + only_b[:5]):
            e = map_a.get(k) or map_b[k]
            print(f"  [p{e['page']}] {_short(e['src'], 70)}")


# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="traducir PDF de referencia y guardar run")
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--pdf", default=None)
    p_run.add_argument("--pages", default=None, help='ej. "1-20" o "1,5,30-40"')
    p_run.add_argument("--src", default="en")
    p_run.add_argument("--tgt", default="es")
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff", help="comparar dos runs")
    p_diff.add_argument("label_a")
    p_diff.add_argument("label_b")
    p_diff.set_defaults(func=cmd_diff)

    p_scan = sub.add_parser("scan", help="bloques sospechosos de un run")
    p_scan.add_argument("label")
    p_scan.add_argument("--show", type=int, default=40)
    p_scan.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
