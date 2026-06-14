# TraductorPDF

> Traductor de PDF 100% local y offline (EN↔ES) que preserva el layout: imágenes, tablas, tipografía y posición del texto. Sin APIs externas, sin cuentas, sin IA en la nube.

**Este README documenta la arquitectura y la hoja de ruta del proyecto.** Si solo querés instalarlo y usarlo, andá directo a [Instalación y uso](#instalación-y-uso).

## Diagnóstico del estado actual

El proyecto YA tiene la forma arquitectónica correcta —un pipeline de tres etapas `extractor.py` → `translator.py` → `builder.py`— que coincide exactamente con el patrón Frontend/Midend/Backend al que convergieron BabelDOC y PDFMathTranslate (pdf2zh v2). Eso es lo más importante: la decomposición de base es sólida y está validada por las dos herramientas líderes del campo. El problema no es el esqueleto; es que las RESPONSABILIDADES se filtran entre etapas y el contrato entre ellas es implícito.

**Fortalezas concretas (fundamentadas en el código):**

- **Backbone de traducción acertado.** OPUS-MT `opus-mt-tc-big-en-es` sobre CTranslate2 int8 (`translator.py:21-24, 483-485`) es la elección correcta para EN↔ES en CPU: ~700 tok/s en ~516 MB de RAM, licencia limpia (modelo CC-BY-4.0, motor MIT), y al ser NMT a nivel de oración tiene riesgo de looping casi nulo comparado con un LLM generativo. No hay nada que cambiar en la decisión del motor base.
- **Manejo de longitud español-más-largo ya pensado.** `builder.py` no hace reemplazo 1:1 ingenuo: tiene extensión de bbox lateral (`_compute_x_limits`), extensión vertical acotada (`_compute_max_y`, `_candidate_rects`), reducción de interlineado antes que de fuente, y shrink geométrico de fuente como último recurso (`_plan_fit:233-269`). Esto es precisamente el "adaptive typesetting / localized scaling factor" de BabelDOC, implementado a mano. Es un activo real.
- **Preservación-por-omisión de elementos no traducibles.** Fórmulas, URLs y tablas se detectan y se SALTAN en el extractor para que el builder no redacte esa región y los píxeles originales sobrevivan (`extractor.py:238-240, 268-269`). Conceptualmente es el masking de BabelDOC.
- **Pass de planificación en scratch page con métricas idénticas** (`builder.py:337-380`) y **unificación de tamaño por clase al percentil 25** (`builder.py:389-405`): esto evita el mosaico de tamaños por-bloque que afea una página. Es ingeniería de calidad, no improvisación.
- **`eval.py` es un golden-harness hecho a mano** (run/diff/scan por clave página+bbox, detección de `⁇`, palabras duplicadas, ratios de longitud anómalos). Ya tienes la parte difícil del testing de regresión.

**Debilidades concretas:**

1. **El refactor del `Engine` está roto, no solo "sin cablear".** Esto es más grave de lo planteado. `gui.py:212` hace `src_sp, tgt_sp, ct2_translator = tr.load_model(...)`, pero `load_model` (`translator.py:519-521`) hoy devuelve un objeto `Engine` ÚNICO, no la 3-tupla `(src_sp, tgt_sp, ct2)`. `Engine` no define `__iter__`, así que ese desempaquetado **falla en runtime**. Y aunque no fallara, `translate_batch` (`translator.py:790`) recibe `src_sp, tgt_sp, translator` por separado y llama directo a `src_sp.encode(...)` / `translator.translate_batch(...)` — **nunca toca `Engine.encode()` ni `Engine.translate()`**. Conclusión: NLLB es código muerto inalcanzable desde el flujo real, y la abstracción a medias dejó el camino OPUS en estado inconsistente. Es la deuda #1.

2. **El contrato entre etapas es un `list[list[dict]]` sin tipar** (`extractor.py:219`). Las decisiones de clasificación se TOMAN y se TIRAN: el extractor decide "esto es fórmula/URL/tabla, lo salto" (`extractor.py:268-276`) pero no deja constancia en ningún modelo. Resultado: el builder re-deriva desde cero alineación (`_detect_align`), fuente (`_select_font`) y obstáculos (`_visual_obstacles`), recomputando información que el extractor ya conocía.

3. **Normalización Unicode duplicada y divergente.** `builder.py:12-15` tiene `_FONT_SAFE_MAP`; `translator.py:90-124` tiene `_UNICODE_MAP` + `_DIACRITIC_MAP`. Dos hogares para la misma preocupación, con riesgo de divergir (de hecho ya difieren en cobertura).

4. **`translator.py` es un monolito de 1123 líneas** donde conviven el motor, la segmentación de oraciones, ~15 heurísticas de passthrough (nombres propios, listas de autores, TOC, footers, cross-refs, prefijos de bibliografía) y un post-procesado anti-loop muy pesado (`_truncate_output`, `_collapse_repeated_ngrams`, `_compact_heading`, `_recover_unk_heads`). Todo ese anti-loop es esencialmente un parche contra la fragilidad de OPUS-MT con inputs cortos. Funciona, pero está acoplado al modelo y es difícil de testear en aislamiento.

5. **El glosario es soft, no determinista.** `glossary.lookup` (`glossary.py:287`) solo acierta cuando el bloque COMPLETO coincide con una entrada. Un término técnico DENTRO de un párrafo ("the firewall configuration...") no se toca: queda a merced de lo que el modelo decida. Para terminología en prosa no hay garantía.

---

## Arquitectura objetivo

La idea central, confirmada por las dos herramientas líderes y por el paper de BabelDOC (arXiv 2605.10845, EMNLP 2025), es un **Modelo de Documento intermedio (IR)** que desacopla los metadatos visuales de layout del contenido semántico, y que cada etapa ENRIQUECE en lugar de re-derivar. No es teoría: es el patrón al que pdf2zh v2 migró adoptando BabelDOC como backend.

Para un proyecto de este tamaño la versión correcta del IR es una de dataclasses tipadas —NO el aparato completo de BabelDOC (esquemas RELAX NG/XSD, serialización XML, `il_version_N`). Eso resuelve un problema multi-formato y multi-equipo que tú no tienes.

```
                              ┌─────────────────────────────────────┐
                              │   GLOSARIO / TERMINOLOGÍA (datos)    │
                              │  CSV/TBX externo · términos+forbidden│
                              └───────────────┬─────────────────────┘
                                              │ (se inyecta en TRANSLATE)
                                              ▼
  PDF ──►  EXTRACT ───►  LAYOUT-ANALYSIS ──►  TRANSLATE  ───►  TYPESET+RENDER  ──►  PDF_ES
           (frontend)      (clasifica)         (midend)          (backend)
              │                 │                  │                  │
              │ parse nativo    │ marca block.kind │ solo body/      │ redact +
              │ get_text(       │ {body,heading,   │  heading/toc/    │ insert con
              │  'rawdict')     │  toc,formula,    │  caption         │ auto-fit
              │ de-hyphen       │  url,table,      │ mask términos    │ (scale loop)
              │ spans+bbox      │  passthrough}    │ + restore        │ font embed
              ▼                 ▼  + obstacles      ▼  determinista    ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                    DOCUMENT  (IR tipado)                          │
        │   Document(source, lang_pair, pages: list[Page])                  │
        │   Page(blocks: list[Block], obstacles: list[Rect])               │
        │   Block(bbox, spans, kind, indent, style, text, translated|None) │
        └──────────────────────────────────────────────────────────────────┘
                          ▲ TODAS las etapas leen/escriben EL MISMO objeto
```

**Diferencias clave frente al estado actual:**

- **`Block.kind` reemplaza el "saltar y olvidar".** Hoy `extractor.py` omite fórmulas/URLs/tablas; mañana las MARCA (`kind="formula"`). La etapa TRANSLATE se vuelve trivial: traduce donde `kind in {body, heading, toc, caption}`, pasa el resto sin tocar. El builder lee `block.kind` en vez de re-clasificar.
- **Layout-analysis es su propia etapa etiquetada**, pero sigue siendo HEURÍSTICA (PyMuPDF `find_tables`, `get_drawings`, `get_images`, `_looks_like_toc_block`). NO traer DocLayout-YOLO ahora (ver decisión b). El valor de aislarla es que su salida —el `kind` y los obstáculos— queda grabada en el IR, y queda una costura limpia por si algún día se quiere meter un modelo.
- **El glosario/terminología es un dato externo y una etapa propia**, no lógica interna del traductor. Corre sobre el IR antes de TRANSLATE y aplica masking determinista (ver decisión c).
- **El motor se aísla tras UN protocolo `Translator`** (la única abstracción que se gana el sueldo, porque YA soportas dos backends). El resto del pipeline nunca sabe si corre OPUS o NLLB.

Lo que NO se debe hacer: ports/adapters/use-cases/DI alrededor de PyMuPDF o de la GUI. PyMuPDF nunca se va a cambiar y sus coordenadas permean todo el render; abstraerlo sería over-engineering de manual para una herramienta solo-dev.

---

## Decisiones clave (tradeoffs + recomendación)

### (a) Modelo: OPUS-MT vs NLLB-200 vs Argos vs LLM local pequeño

| Opción | Pros | Contras |
|---|---|---|
| **OPUS-MT tc-big** | Mejor calidad/MB en EN↔ES (FLORES BLEU 28.5, Tatoeba 57.2), ~700 tok/s CPU, ~516 MB, looping casi nulo (NMT por oración), licencia limpia | Sin contexto cross-oración; un modelo por dirección |
| NLLB-200 distilled | Cobertura multilingüe en un modelo | Más pesado, **sin ventaja** en EN↔ES (gasta capacidad en 200 idiomas), licencia **CC-BY-NC** (no comercial), over-generación/repetición documentada |
| Argos Translate | Offline llave-en-mano (CTranslate2+SentencePiece+Stanza), MIT/CC0 | Es esencialmente "OPUS-MT empaquetado" con checkpoints más viejos; menos control de decoding |
| LLM local (Gemma2-9B/Qwen2.5-7B) | Techo de fluidez/idiom en alta-recurso | Mucho más lento y RAM-hambriento en CPU; repetición/looping se dispara 5-50× por debajo de ~7B y en Q4; puede reescribir/perder contenido técnico |

**Recomendación: mantener OPUS-MT tc-big como motor por defecto. Punto.** Es la decisión correcta y ya está tomada en el código. NLLB queda como opción secundaria SOLO si algún día se necesitan más idiomas (y aceptando la licencia NC) — pero primero hay que CABLEARLO de verdad vía el protocolo `Translator`, porque hoy es código muerto. El LLM local NO entra como motor masivo: a lo sumo, mucho más adelante, como segundo pase selectivo sobre pasajes literarios concretos, a Q8 y con un guard de looping. Para un traductor de libros técnicos EN↔ES, gastar CPU en un LLM es pagar de más sin retorno en este par de alta recurso.

### (b) Tablas/figuras/fórmulas: saltar-como-píxeles vs modelo de layout-analysis

| Opción | Pros | Contras |
|---|---|---|
| **Saltar-como-píxeles (actual)** | Cero dependencias ML, CPU-barato, las tablas/figuras renderizan idénticas al original, ya implementado | Precisión heurística; falla en tablas sin bordes o fórmulas inline sutiles |
| DocLayout-YOLO (ONNX) | Precisión SOTA, clasifica 10 clases con captions separados | Dependencia ONNX/torch + gestión de pesos; **licencia AGPL-3.0** (copyleft, sin escape comercial libre — necesita licencia Enterprise de Ultralytics) |
| Surya 2 / PP-Structure | OCR + layout + reading-order en un modelo | Surya pesado y lento en CPU + pesos con cap de ingresos; PP-Structure es Apache pero arrastra runtime PaddlePaddle |

**Recomendación: quedarse con la detección heurística de PyMuPDF, formalizada como etapa que estampa `block.kind`.** Para libros digitales (no escaneados) EN↔ES, las heurísticas actuales (`find_tables`, `get_drawings`, `_is_formula_block`) cubren la gran mayoría de casos sin meter una dependencia ML pesada justificada solo para los targets de precisión de papers científicos. **Importante sobre licencias:** DocLayout-YOLO NO es permisivo —es AGPL-3.0, igual que PyMuPDF—. La verdad incómoda: PyMuPDF ya te ata a AGPL/comercial. Mientras la herramienta sea offline y de uso interno/personal, AGPL no te obliga a nada; si algún día la distribuyes como producto cerrado o SaaS, hay que resolver PyMuPDF (licencia Artifex, o swap a pypdfium2/pdfminer.six) ANTES de añadir un segundo componente copyleft. Diseña la costura del IR para poder enchufar un modelo de layout detrás de `block.kind` más adelante; defiere el modelo.

### (c) Inyección de terminología: placeholder-masking vs post-replace vs constrained decoding

El número que decide es duro: en la WMT 2023 Terminology Shared Task, **ningún sistema neural superó ~60-70%** de aciertos de terminología, incluso con el diccionario provisto. **Ningún método puramente neural es determinista.**

| Técnica | Garantía | Problema |
|---|---|---|
| **Placeholder masking + restore** | **100% determinista** | El modelo no ve el significado → puede romper concordancia/género/orden alrededor |
| Post-replace por alineación | Superficialmente determinista | Errores de alineación + morfología → reemplazos no gramaticales (el más débil para ES) |
| Constrained/forced decoding | Garantiza la cadena | Caro, frágil, **menos fluido para lenguas con flexión rica como el español** (fuerza la forma de superficie ignorando concordancia), post-proceso pesado |
| Soft-constraint (Dinu 2019, OPUS-CAT) | Probabilístico (~60-70%) | Requiere fine-tuning; no garantiza |

**Recomendación: capa determinista de diccionario por masking + restore verificado, con OPUS-MT como fallback de fluidez.** El glosario actual (match de bloque completo) hay que evolucionarlo a: (1) lookup por longest-match sobre términos normalizados DENTRO del texto, (2) reemplazar cada término por un centinela que OPUS-MT pase intacto, (3) traducir, (4) restaurar el target curado, (5) verificar que aterrizó. Como el español es morfológicamente rico, NO confíes solo en restaurar una forma cruda: guarda variantes flexionadas/género en el termbase, o aplica un paso ligero de flexión al restaurar — forzar una forma sin flexionar es exactamente lo que vuelve agramatical al hard-constraint. Mover el glosario a un CSV/TBX externo y auditable (semilla: Microsoft Terminology Collection, TBX gratis con español) con tipos `do-not-translate`/`forbidden` para marcas y acrónimos. NO copies el enfoque de pdf2zh (glosario inyectado en prompt de LLM): es soft y, por defecto, online.

### (d) Estrategia de reflow para "ES es más largo"

El español corre ~15-30% más largo (promedio ~25%; cadenas cortas hasta 200-300%). El reflow 1:1 es imposible; hay que reflow + shrink. El builder ya hace casi todo esto bien.

**Recomendación: conservar el algoritmo de typeset actual (extensión de bbox → interlineado → shrink de fuente) y migrar de `insert_textbox` a `insert_htmlbox`.** `insert_htmlbox(rect, html, scale_low=...)` devuelve `(spare_height, scale)` y auto-reduce para encajar (`spare_height == -1` señala fallo de ajuste), además de permitir llevar spans bold/italic/color como CSS inline desde el IR, lo que `insert_textbox` no puede. Reserva `insert_textbox` + loop manual `fontsize -= 0.5` para labels de una línea. Fija un piso de legibilidad (`scale_low ~0.6-0.7`) y registra los bloques que lo tocan para revisión manual — exactamente lo que `eval.py` ya empieza a hacer. El orden de prioridad del builder actual (crecer lateral antes que encoger fuente) es el correcto porque preserva el tamaño donde crecer hacia abajo invadiría el párrafo siguiente.

---

## Plan de migración incremental

Sin big-bang. Cada paso es independientemente entregable y testeable.

**Fase 0 — Arreglar el bug del Engine (bloqueante).** Decidir UNA firma. O `load_engine` devuelve `Engine` y `translate_batch` recibe `engine` (y usa `engine.encode/translate`), o se elimina `Engine` y se vuelve a la 3-tupla limpia. Hoy el código está en un estado intermedio que rompe el desempaquetado en `gui.py:212`. Esto se hace ANTES de cualquier refactor de IR.

**Fase 1 — Dataclasses tipadas (envoltura fina).** Crear `model.py` con `Block`/`Page`/`Document`. `extract_blocks` devuelve `Document`; `build_translated_pdf` lee `Document`. Comportamiento idéntico, solo se tipa el contrato. ~50 líneas.

**Fase 2 — Grabar clasificación en `block.kind`.** Mover las decisiones formula/URL/TOC/table del extractor a `block.kind`. Borrar la re-derivación del builder (`_detect_align`/`_select_font` leen del IR). Esto mata la duplicación de `_FONT_SAFE_MAP` vs `_UNICODE_MAP` dándoles un solo hogar.

**Fase 3 — Cristalizar el protocolo `Translator`.** `Translator.translate(texts, src, tgt) -> list[str]` con `OpusTranslator` y `NllbTranslator`. Mover batching, heurísticas de passthrough y post-proceso anti-loop a un wrapper delgado; dejar solo la llamada al modelo abstracta. Recién aquí NLLB queda realmente alcanzable.

**Fase 4 — Glosario como etapa + datos externos.** Sacar `glossary.py` a CSV/TBX externo. Implementar masking determinista in-text. Etapa propia antes de TRANSLATE.

**Fase 5 — Promover `eval.py` a suite golden** (pytest-golden o pytest-regressions): fixtures PDF pequeños, JSON dorado por bloque, `--update-goldens` para re-bless, y los heurísticos de `scan` (`⁇`, dup-words, ratios, trigramas en loop) como ASSERTS que rompen el build. Dos tiers: estructural determinista (bbox/kind/fit, match exacto) y calidad de traducción (tolerancia). Se hace AL FINAL para fijar el refactor.

**Trampas de over-engineering a EVITAR explícitamente:**
- NO esquemas RELAX NG/XSD ni serialización XML ni `il_version_N` para el IR. Dataclasses y ya.
- NO arquitectura hexagonal / ports-adapters / DI containers. Tienes UN eje real para abstraer (el motor) y uno que nunca cambiarás (PyMuPDF).
- NO traer DocLayout-YOLO/Surya/MinerU ahora. Diseña la costura, defiere el modelo.
- NO un LLM local como motor masivo.
- NO vendorizar BabelDOC/PDFMathTranslate (AGPL): copia el PATRÓN, no el código.
- NO diseñar el IR completo de antemano: empieza por la envoltura fina y deja que evolucione.

---

## Quick wins

Las 3-5 primeras intervenciones con mejor relación valor/esfuerzo:

1. **Arreglar el desfase `Engine`/`load_model`** (Fase 0). Es un bug de runtime real en `gui.py:212` y la causa de que NLLB sea inalcanzable. Esfuerzo: bajo. Valor: desbloquea todo lo demás y evita un crash.

2. **Sacar el glosario a un CSV externo** sin tocar la lógica. `GLOSSARY` (290 líneas hardcoded en `glossary.py`) pasa a `glossary.csv` cargado al inicio. Esfuerzo: ~30 min. Valor: el glosario se vuelve dato auditable y editable sin tocar código —y es el primer paso hacia la etapa de terminología.

3. **Unificar la normalización Unicode en un solo módulo.** Hoy `_FONT_SAFE_MAP` (builder) y `_UNICODE_MAP`/`_DIACRITIC_MAP` (translator) divergen. Un solo `text_norm.py`. Esfuerzo: bajo. Valor: elimina una fuente silenciosa de inconsistencia entre etapas.

4. **Migrar el render de `insert_textbox` a `insert_htmlbox`** para los bloques de body/heading, leyendo su `(spare_height, scale)` para el auto-fit y registrando los que tocan el piso de `scale_low`. Esfuerzo: medio. Valor: mejor ajuste del texto español-más-largo y, de paso, preservación de bold/italic/color vía CSS inline — el mayor salto de fidelidad visual disponible.

5. **Convertir `eval.py` en una suite pytest con 2-3 PDFs fixture** y los heurísticos de `scan` como aserciones. Esfuerzo: medio. Valor: cada bug que arregles añade un caso dorado; bloquea regresiones antes de cada refactor. Es el seguro que hace seguros todos los pasos anteriores.

---

## Instalación y uso

> Guía para usuarios finales. La arquitectura y la hoja de ruta están arriba.

### Características

- **Sin internet después del setup** — los modelos se descargan una sola vez y quedan en tu máquina
- **Sin cuentas ni suscripciones** — todo corre localmente
- **Preserva el layout** — imágenes, colores y posición del texto intactos
- **Tipografía adaptada** — detecta bold, italic, serif, mono y elige la fuente más cercana
- **Rápido** — motor CTranslate2 int8, ~15-25 min para un libro de 6.000+ bloques en CPU

### Idiomas soportados

| Par | Modelo |
|-----|--------|
| Inglés → Español | Helsinki-NLP/opus-mt-tc-big-en-es |
| Español → Inglés | Helsinki-NLP/opus-mt-es-en |

### Requisitos

- Python 3.10 o superior
- ~400 MB de espacio en disco (modelos)

### Instalación

```bash
pip install pymupdf ctranslate2 sentencepiece huggingface_hub
```

`tkinter` viene incluido con Python.

### Uso

```bash
python main.py
```

1. Clic en **Browse...** y selecciona el PDF
2. Elige los idiomas **From** y **To**
3. Elige la carpeta de destino (por defecto la misma del PDF)
4. Clic en **Translate PDF**

La primera vez que usas un par de idiomas, el modelo se descarga y convierte
automáticamente (~5 min). Las siguientes ejecuciones cargan en ~3-5 segundos.
El PDF traducido se guarda como `nombre_original_translated.pdf`.

### Limitaciones conocidas

- **PDFs escaneados** — si el PDF es una imagen (sin texto seleccionable), se
  requeriría OCR, que no está incluido en esta versión.
- **Tablas y layouts complejos** — el extractor trata cada bloque de forma
  independiente; tablas multicolumna o layouts muy elaborados pueden quedar
  desalineados respecto al original.
- **Solo EN↔ES** — para añadir más pares ver `SUPPORTED_PAIRS` en `translator.py`.

---

## Anexo — Verificación adversarial de afirmaciones clave

Las afirmaciones que sustentan cada decisión fueron verificadas por agentes independientes contra fuentes primarias (papers, READMEs de GitHub, docs). Resultado: **9 confirmadas, 1 parcial** de 10 verificadas. La evidencia cruda completa vive en el run del workflow. Las afirmaciones se conservan en su idioma original de investigación (inglés) para fidelidad de la auditoría.

**✅ Confirmado** — The dominant open-source local PDF-translation pipeline is: layout detection -> split into regions (text/formula/table/figure) -> translate ONLY text strings -> re-render translated text into original coordinates. PDFMathTranslate stages are 'layout detection, splitting…
  - Matiz: The core claim and both quotes are accurate, but "identical across all leading tools" and "translate ONLY text strings" overstate convergence. PDFMathTranslate does translate isolated region text strings, but BabelDOC's 2026 paper (arXiv 2605.10845, "Better Layout-Preserving PDF Translation via Intermediate Representation") explicitly argues the BETTER design adds an Intermediate Representation (IR) layer that decouples visual layout metadata from semantic content, enabling document-level translation (cross-page context, glossary constraints, and especially FORMULA PLACEHOLDERING) rather than per-region string translation. So the shape…
  - Fuentes: https://arxiv.org/html/2507.03009v2 · https://github.com/funstory-ai/BabelDOC/blob/main/README.md · https://arxiv.org/abs/2605.10845

**✅ Confirmado** — The single most valuable BabelDOC architectural idea is an Intermediate Representation that decouples visual layout metadata from semantic text, using explicit placeholders ({v1} for formulas, <style id='1'> for inline rich-text style runs); translation operates on…

**✅ Confirmado** — Translated text changes length, so an 'adaptive typesetting engine' re-fits it into the original bounding box by adjusting localized scaling factors (font size) and handling line breaks. This is the piece that actually preserves the visual layout. (Attributed to BabelDOC arXiv…
  - Matiz: No correction needed; the claim is accurate. One clarifying note (strengthens rather than refutes): the scale-to-fit / adaptive typesetting mechanism is BabelDOC's contribution, while PDFMathTranslate's cited "subset font embedding" + "download_remote_fonts" solve a DIFFERENT problem — embedding correct target-language glyphs (CJK etc.), not text re-fitting. The claim already keeps these separate, which is correct. For a small Python project: the load-bearing layout-preservation logic is the iterative scaling loop (decrement scale by 0.05 when >0.6, 0.10 when <=0.6; reduce line spacing toward min 1.4; fail below 0.1) PLUS sequential element…
  - Fuentes: https://arxiv.org/abs/2605.10845 · https://arxiv.org/html/2605.10845v1 · https://funstory-ai.github.io/BabelDOC/ImplementationDetails/Typesetting/Typesetting/ · https://arxiv.org/html/2507.03009v2

**✅ Confirmado** — Both leading offline layout-preserving PDF translators (PDFMathTranslate and BabelDOC) parse the PDF's native text+coordinate stream (no OCR for digital PDFs). DocLayout-YOLO (a YOLOv10-based detector) classifies page regions; pdfminer.six/custom interpreters extract char-level…
  - Matiz: The core architectural claim is fully confirmed by primary sources. One nuance on "CPU-cheap": pdfminer.six native extraction (no OCR) IS genuinely lightweight, and the system is explicitly CPU-runnable — the arXiv paper states ONNX was "chosen as default to ensure the compatibility of our parsing pipeline for diverse hardwares" and the runtime uses CPUExecutionProvider. However, "CPU-cheap" is a slight overstatement for the layout step specifically: DocLayout-YOLO is a real-time neural detector whose training requires GPUs (8 GPUs in the official repo) and which runs fastest on CUDA; CPU inference works but is the heaviest stage of the…
  - Fuentes: https://github.com/PDFMathTranslate/PDFMathTranslate · https://arxiv.org/html/2507.03009v1 · https://aclanthology.org/2025.emnlp-demos.71/ · https://funstory-ai.github.io/BabelDOC/ImplementationDetails/PDFParsing/PDFParsing/ · https://github.com/opendatalab/DocLayout-YOLO

**✅ Confirmado** — The translation backend is a pluggable middleware. PDFMathTranslate separates 'layout flows from translation flows' so a backend is ~15 lines (inherit base, implement do_translate); supports 23+ services including OFFLINE options Ollama, Xinference, and Argos Translate. BabelDOC…
  - Matiz: One scoping nuance an architect should not miss (load-bearing for tool selection): the 23-service list with Argos Translate belongs to the ORIGINAL PDFMathTranslate (pdf2zh, the EMNLP paper / first-gen repo). The successor line — PDFMathTranslate-next (pdf2zh_next) and BabelDOC, both by funstory-ai — narrowed the surface: BabelDOC is OpenAI-compatible-LLM-ONLY (no Argos, no Google/DeepL/Bing native backends), pushing everything through one LLM interface. The DeepWiki page for PDFMathTranslate-next lists only Ollama, Xinference, and OpenAI-compatible endpoints as local options and does NOT list Argos Translate. So "supports 23+ services…
  - Fuentes: https://arxiv.org/html/2507.03009v2 · https://aclanthology.org/2025.emnlp-demos.71.pdf · https://github.com/funstory-ai/BabelDOC · https://github.com/PDFMathTranslate/PDFMathTranslate · https://deepwiki.com/PDFMathTranslate/PDFMathTranslate-next/4.3-local-and-self-hosted-models

**✅ Confirmado** — Fully-offline CPU translation is achievable with Argos Translate (engine behind LibreTranslate): OpenNMT models run via CTranslate2, packaged as .argosmodel zips, with automatic pivot translation (es->en->fr) for missing pairs. License is MIT OR CC0.
  - Matiz: No correction needed. Three nuances for the project: (a) the .argosmodel zip also bundles SentencePiece and Stanza models, not just CTranslate2 - Stanza adds a non-trivial offline dependency footprint; (b) pivot es->en->fr works but at lower fidelity than a direct pair per official docs; (c) argostranslate itself is MIT/CC0, but LibreTranslate's API server is AGPL-3.0 - the permissive advantage applies to depending on argostranslate directly, not to embedding the LibreTranslate server.
  - Fuentes: https://github.com/argosopentech/argos-translate/

**✅ Confirmado** — Neither PDFMathTranslate nor BabelDOC does OCR; they cannot handle scanned PDFs. This is an explicit, stated limitation. For scanned input you must bolt on an OCR/VLM front-end (Surya/marker, MinerU).
  - Matiz: The core claim is correct, but one nuance must be stated precisely so it does not mislead an architecture decision: BabelDOC ships an `--ocr-workaround` flag (and `--auto-enable-ocr-workaround`), which can give the FALSE impression that BabelDOC does OCR. It does NOT. The flag is a cosmetic typesetting step: it draws white rectangles below the translated text to cover the original and forces all text to black, and it only works for documents that ALREADY have a text layer with black-text-on-white-background. It does not extract text from images. So the accurate statement is: neither tool performs optical character recognition; both require a…
  - Fuentes: https://arxiv.org/html/2507.03009v2 · https://github.com/funstory-ai/BabelDOC/blob/main/README.md · https://aclanthology.org/2025.emnlp-demos.71/ · https://pypi.org/project/BabelDOC/

**✅ Confirmado** — Surya v2 (2025-2026) is a single ~650M-param vision-language model doing OCR, layout, reading-order and table recognition, emitting a per-block 'reading_order' index. marker layers heuristics + Surya + Texify (LaTeX) to produce Markdown/JSON. Both run offline on CPU via…

**✅ Confirmado** — marker is NOT a translator — it is a PDF->Markdown/JSON converter. To translate while preserving layout you would translate its structured output and re-render yourself, or use it only as the extraction/OCR front-end feeding a separate render step.
  - Fuentes: https://github.com/datalab-to/marker/blob/master/README.md

**🟡 Parcial** — Licensing is a real architectural constraint for offline layout-preserving PDF translation. PDFMathTranslate and BabelDOC are AGPL-3.0. marker code is GPL-3.0 and Surya/marker model weights use a modified 'AI Pubs Open Rail-M' license (free only under revenue/funding caps: Surya…
  - Matiz: All individual license facts are CONFIRMED, but the framing of the "permissive offline path" is misleading on one component. DocLayout-YOLO is NOT permissive — it is AGPL-3.0 (copyleft, source-disclosure), because it is built on Ultralytics + YOLOv10, both AGPL-3.0. So the proposed "permissive" stack actually contains TWO copyleft pieces, not zero: PyMuPDF (AGPL, but with a commercial escape hatch from Artifex) AND DocLayout-YOLO (AGPL, with NO free commercial escape — needs an Ultralytics Enterprise License to avoid open-sourcing your whole app). Only Argos (MIT/CC0) is genuinely permissive. Architectural correction: if you want a truly…
  - Fuentes: https://github.com/datalab-to/marker · https://github.com/datalab-to/surya · https://github.com/Byaidu/PDFMathTranslate · https://github.com/funstory-ai/BabelDOC · https://github.com/argosopentech/argos-translate · https://github.com/pymupdf/PyMuPDF

