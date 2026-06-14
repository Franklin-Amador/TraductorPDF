"""Genera el PDF traducido completo (pipeline entero).

Uso: python _run_full.py [ruta_pdf] [sufijo]
Sin argumentos usa el PDF de referencia DDIA.
"""
import os
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

import extractor
import translator
import builder

DEFAULT_PDF = (
    r"C:\Users\joela\Downloads"
    r"\Designing Data-Intensive Applications, 2nd Edition-1-55.pdf"
)

pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
suffix = sys.argv[2] if len(sys.argv) > 2 else "translated_fase1"
base, _ = os.path.splitext(pdf)
out = f"{base}_{suffix}.pdf"

t0 = time.time()
print(f"Extrayendo bloques de: {os.path.basename(pdf)}")
all_pages = extractor.extract_blocks(pdf)
flat_texts = [b["text"] for page in all_pages for b in page]
print(f"  {len(all_pages)} páginas, {len(flat_texts)} bloques")

print("Cargando modelo en→es…")
src_sp, tgt_sp, ct2 = translator.load_model("en", "es")

last = -1
def progress(done, total):
    global last
    pct = done * 100 // total
    if pct != last:
        last = pct
        print(f"\r  Traduciendo… {pct}%", end="", flush=True)

translated_flat = translator.translate_batch(flat_texts, src_sp, tgt_sp, ct2, progress)
print()

print("Reconstruyendo PDF…")
translated_pages = []
idx = 0
for page in all_pages:
    translated_pages.append(translated_flat[idx: idx + len(page)])
    idx += len(page)

builder.build_translated_pdf(pdf, all_pages, translated_pages, out)
print(f"Listo en {time.time() - t0:.0f}s → {out}")
