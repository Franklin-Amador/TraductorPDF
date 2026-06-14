# PDF Translator

Traductor de PDFs 100% local, gratuito y sin cuentas. Preserva el layout, imágenes y
tipografía del original. Funciona completamente offline después de la descarga inicial
del modelo de traducción.

---

## Características

- **Sin internet después del setup** — los modelos se descargan una sola vez y quedan en tu máquina
- **Sin cuentas ni suscripciones** — todo corre localmente
- **Preserva el layout** — imágenes, colores y posición del texto intactos
- **Tipografía adaptada** — detecta bold, italic, serif, mono y elige la fuente más cercana
- **Rápido** — motor CTranslate2 int8, ~15-25 min para un libro de 6.000+ bloques en CPU

---

## Idiomas soportados

| Par | Modelo |
|-----|--------|
| Inglés → Español | Helsinki-NLP/opus-mt-tc-big-en-es |
| Español → Inglés | Helsinki-NLP/opus-mt-es-en |

---

## Requisitos

- Python 3.10 o superior
- ~400 MB de espacio en disco (modelos)

---

## Instalación

```bash
pip install pymupdf ctranslate2 sentencepiece huggingface_hub
```

No se necesita ninguna otra dependencia. `tkinter` viene incluido con Python.

---

## Uso

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

---

## Estructura del proyecto

```
Traductor/
├── main.py        # Punto de entrada
├── gui.py         # Interfaz Tkinter
├── extractor.py   # Extrae bloques de texto del PDF
├── translator.py  # Motor de traducción local (CTranslate2)
├── builder.py     # Reconstruye el PDF con el texto traducido
├── models_ct2/    # Modelos convertidos (se crea automáticamente)
├── AGENTS.md      # Documentación técnica para agentes de IA
└── README.md      # Este archivo
```

---

## Limitaciones conocidas

- **PDFs escaneados** — si el PDF es una imagen (no tiene texto seleccionable), el
  traductor no puede extraer texto. Se requeriría OCR, que no está incluido en esta versión.

- **Nombres propios** — nombres de personas y marcas en portadas o créditos pueden
  aparecer ligeramente distorsionados. El contenido técnico (párrafos, capítulos) se
  traduce bien.

- **Tablas y layouts complejos** — el extractor trata cada bloque de texto de forma
  independiente. Tablas con múltiples columnas o layouts muy elaborados pueden quedar
  con el texto desalineado respecto al original.

- **Solo EN↔ES** — para añadir más idiomas ver `SUPPORTED_PAIRS` en `translator.py`.

---

## Cómo funciona internamente

1. **Extracción** (`extractor.py`) — PyMuPDF lee el PDF y devuelve cada bloque de texto
   con sus coordenadas exactas (bounding box) y metadatos de fuente.

2. **Traducción** (`translator.py`) — cada bloque se divide en oraciones individuales,
   se normalizan los caracteres Unicode especiales, se tokenizan con SentencePiece y se
   traducen en batches ordenados por longitud. Un post-proceso elimina las continuaciones
   espurias que el modelo a veces genera.

3. **Reconstrucción** (`builder.py`) — PyMuPDF redacta (borra) el texto original de cada
   área usando redacciones PDF y luego reinserta la traducción en el mismo lugar, con la
   fuente, tamaño y color más cercanos al original. Si el texto traducido no cabe, el
   tamaño de fuente se reduce progresivamente.
