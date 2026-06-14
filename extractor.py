import re
import statistics
import pymupdf

# A TOC line ends with whitespace-padded page number ("...   43" or "... . . . 43").
# Used to detect whether a block is a Table-of-Contents block whose lines should
# each become an independent block (each gets its own bbox so the builder can
# reinsert the translation in the correct line position).
_TOC_LINE_RE = re.compile(r'[\s.]{3,}[\divxlcdm]+\s*$', re.IGNORECASE)

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")

# Hyphens that may appear at a line break: ASCII, U+2010 hyphen, U+2011 nb-hyphen
_LINE_HYPHENS = ("-", "‐", "‑")

# URL / domain inside a short block (a caption or heading that is just a link).
_URL_RE = re.compile(r"(?:https?://|www\.)\S|\b\S+\.(?:com|org|net|ai|io|dev|edu|gov)\b", re.I)

# Math indicators: operators, or a single letter applied like a function "P (".
# (No "|": it appears in running-header footers "Summary | 25", not just math.)
_MATH_RE = re.compile(r"[=∏∑∫√≤≥≈≠×÷·∈∉⊂→←↔∀∃∇∂±∞]|[A-Za-z]\s*\(\s*[A-Za-z0-9]")

# Dot leader of a table-of-contents entry ("Title . . . . 43").
_DOTLEADER_RE = re.compile(r"(?:\.\s*){4,}")


def _is_url_block(text: str) -> bool:
    """A short block that is essentially a URL/domain (e.g. a footer link or a
    heading that is only an address).  Translating it mangles it ("www" →
    "w w w"); leave the original pixels untouched."""
    return len(text) < 90 and _URL_RE.search(text) is not None


def _is_formula_block(text: str) -> bool:
    """A block dominated by math notation rather than prose.  These are short,
    have few real words and many single-character / operator tokens; the model
    garbles them ("P ( x 1 , x 2 )" → soup).  Leave the original pixels.
    """
    if len(text) > 160:
        return False
    # TOC entries ("Preface . . . . xvii") are dot-heavy but not math.
    if _DOTLEADER_RE.search(text):
        return False
    tokens = text.split()
    if len(tokens) < 3:
        return False
    words = sum(1 for t in tokens if t.isalpha() and len(t) >= 3)
    singles = sum(1 for t in tokens if len(t.strip(".,;:()[]{}")) <= 1)
    word_ratio = words / len(tokens)
    single_ratio = singles / len(tokens)
    has_math = _MATH_RE.search(text) is not None
    return word_ratio < 0.35 and (single_ratio > 0.4 or has_math)


def _table_rects(page) -> list:
    """Bounding boxes of real tables (≥2×2) on the page.  Blocks inside them
    are left untouched so the table renders exactly as the original — the model
    otherwise reflows the cells into overlapping soup and erases colored header
    rows."""
    rects = []
    try:
        for tb in page.find_tables().tables:
            if tb.col_count >= 2 and tb.row_count >= 2:
                rects.append(pymupdf.Rect(tb.bbox))
    except Exception:
        pass
    return rects


def _in_any(bbox, rects) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in rects)


def _collect_vocab(doc) -> set[str]:
    """All lowercase words in the document.  Used to decide whether a
    line-break hyphen is syllabification ("informa-/tion" → "information",
    which appears elsewhere in the text) or a real compound
    ("analysis-/friendly" → keep the hyphen).
    """
    vocab: set[str] = set()
    for page in doc:
        for w in page.get_text("words"):
            for m in _WORD_RE.finditer(w[4]):
                vocab.add(m.group(0).lower())
    return vocab


def _join_lines(line_texts: list[str], vocab: set[str]) -> str:
    """Join the lines of a block, repairing words split by end-of-line hyphens.

    Without this, "informa-\\ntion" reaches the model as "informa- tion"
    (broken tokens → degraded translation).  Rules:
      • merged word exists in the document vocabulary → drop the hyphen
        ("informa-/tion" → "information")
      • typographic hyphen U+2010/U+2011 + lowercase continuation → drop it:
        professional typesetting uses it for syllabification ("Hernan‐/dez"
        → "Hernandez"), unlike ASCII "-" which marks real compounds
      • any other hyphen → join without the spurious space
        ("analysis-friendly", "TCP-IP")
      • no hyphen → join with a single space (original behavior)
    """
    text = ""
    for lt in line_texts:
        if not text:
            text = lt
            continue
        if text.endswith(_LINE_HYPHENS) and lt and lt[0].isalpha():
            head = text[:-1]
            tail_m = re.search(r"[A-Za-zÀ-ÿ]+$", head)
            next_m = _WORD_RE.match(lt)
            merged_in_vocab = (
                tail_m and next_m
                and (tail_m.group(0) + next_m.group(0)).lower() in vocab
            )
            is_typographic = text[-1] in "‐‑"
            if lt[0].islower() and (merged_in_vocab or is_typographic):
                text = head + lt
            else:
                text = text + lt
        else:
            text = text + " " + lt
    return text


def _merge_singles(singles: list, vocab: set[str], page_index: int) -> list[dict]:
    """Merge a run of consecutive ONE-LINE blocks into paragraphs.

    Some PDFs (esp. fiction) emit one block per visual line.  Translating each
    line independently shreds grammar ("...smoke and" / "sulfur." become two
    fragments) and scatters font sizes.  Paragraphs here are delimited only by
    a first-line indent (no blank line between them), so the rule is:

      • a line indented past the column margin STARTS a new paragraph
      • a much larger-than-normal vertical gap STARTS a new paragraph
        (covers books that separate paragraphs with blank space instead)
      • otherwise the line continues the current paragraph

    Multi-line blocks (already-formed paragraphs, e.g. the technical book) never
    reach this function, so its behavior is unchanged for them.

    A single-line paragraph keeps its original bbox (so a centered title or an
    isolated indented line stays put).  A merged multi-line paragraph is
    left-aligned to the column margin and records its first-line indent in
    points (`indent`) for the builder to reproduce.
    """
    if not singles:
        return []

    margin = min(s["bbox"][0] for s in singles)
    gaps = [b["bbox"][1] - a["bbox"][3] for a, b in zip(singles, singles[1:])]
    leading = statistics.median(gaps) if gaps else 0.0
    INDENT_MIN = 6.0
    gap_break = max(leading * 1.8, leading + 5.0)

    paras: list[list[dict]] = []
    cur: list[dict] | None = None
    for s in singles:
        x0 = s["bbox"][0]
        indented = x0 > margin + INDENT_MIN
        start_new = cur is None
        if cur is not None:
            gap = s["bbox"][1] - cur[-1]["bbox"][3]
            prev_x1 = max(l["bbox"][2] for l in cur)
            x_overlap = min(s["bbox"][2], prev_x1) - max(s["bbox"][0],
                            min(l["bbox"][0] for l in cur))
            if indented or gap > gap_break or x_overlap < 0:
                start_new = True
        if start_new:
            cur = [s]
            paras.append(cur)
        else:
            cur.append(s)

    out: list[dict] = []
    for p in paras:
        if len(p) == 1:
            # Keep original bbox (centered titles / isolated indented lines)
            out.append({
                "bbox": p[0]["bbox"],
                "text": p[0]["text"],
                "spans": p[0]["spans"],
                "page_index": page_index,
                "indent": 0.0,
            })
            continue
        x0 = margin
        y0 = min(l["bbox"][1] for l in p)
        x1 = max(l["bbox"][2] for l in p)
        y1 = max(l["bbox"][3] for l in p)
        text = _join_lines([l["text"] for l in p], vocab)
        indent_pts = max(0.0, p[0]["bbox"][0] - margin)
        out.append({
            "bbox": (x0, y0, x1, y1),
            "text": text,
            "spans": p[0]["spans"],
            "page_index": page_index,
            "indent": indent_pts,
        })
    return out


def _looks_like_toc_block(lines: list) -> bool:
    """A block looks like TOC when ≥3 lines and ≥60% match the TOC-line pattern.
    The threshold tolerates section headings ("Fault Tolerance") mixed with
    sub-entries ("Hardware and Software Faults                  44").
    """
    if len(lines) < 3:
        return False
    matches = 0
    for line in lines:
        text = "".join(s["text"] for s in line["spans"])
        if _TOC_LINE_RE.search(text):
            matches += 1
    return matches >= len(lines) * 0.6


def extract_blocks(pdf_path: str) -> list[list[dict]]:
    doc = pymupdf.open(pdf_path)
    vocab = _collect_vocab(doc)
    all_pages = []
    for page_index, page in enumerate(doc):
        raw = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        table_rects = _table_rects(page)

        # First pass: classify every body block in reading order.
        #   "toc"    – multi-line table-of-contents block (split per line)
        #   "tocln"  – single line that is itself a TOC entry (dots + page num)
        #   "multi"  – already-formed multi-line paragraph (e.g. technical book)
        #   "single" – one visual line, candidate for paragraph merging
        items: list[tuple] = []
        n_single = n_multi = 0
        for block in raw["blocks"]:
            if block["type"] != 0:
                continue

            # Leave tables exactly as the original: skipping their blocks means
            # the builder never redacts that region, so the pixels survive.
            if _in_any(block["bbox"], table_rects):
                continue

            lines = block["lines"]

            if _looks_like_toc_block(lines):
                items.append(("toc", block))
                continue

            spans = [s for line in lines for s in line["spans"]]
            line_texts = [
                " ".join(s["text"].strip() for s in line["spans"]
                         if s["text"].strip())
                for line in lines
            ]
            text = _join_lines([lt for lt in line_texts if lt], vocab)
            if len(text) < 2:
                continue

            if len(lines) == 1:
                raw_line = "".join(s["text"] for s in spans)
                if _TOC_LINE_RE.search(raw_line):
                    items.append(("tocln", block, raw_line.strip(), spans))
                    continue
                # Formula blocks and URL-only captions are left as original
                # pixels: the model turns "P ( x 1 , x 2 )" and "www.x.com" into
                # garbage and reinserting reflows them.  (Checked after TOC so
                # dot-leader entries aren't mistaken for math.)
                if _is_formula_block(text) or _is_url_block(text):
                    continue
                items.append(("single", block, text, spans))
                n_single += 1
            else:
                if _is_formula_block(text) or _is_url_block(text):
                    continue
                items.append(("multi", block, text, spans))
                n_multi += 1

        # Merge single lines into paragraphs only on pages clearly typeset as
        # one-block-per-line (fiction).  On mixed/technical pages the few
        # single-line blocks are headings/captions and must stay standalone.
        merge_mode = n_single >= 4 and n_single >= (n_single + n_multi) * 0.6

        page_blocks: list[dict] = []
        singles: list[dict] = []

        def _flush_singles():
            if singles:
                page_blocks.extend(_merge_singles(singles, vocab, page_index))
                singles.clear()

        for item in items:
            kind = item[0]
            if kind == "single" and merge_mode:
                _, block, text, spans = item
                singles.append({"bbox": block["bbox"], "text": text,
                                "spans": spans})
                continue
            _flush_singles()
            if kind == "toc":
                block = item[1]
                for line in block["lines"]:
                    line_spans = line["spans"]
                    t = "".join(s["text"] for s in line_spans).strip()
                    if len(t) < 2:
                        continue
                    page_blocks.append({
                        "bbox": line["bbox"], "text": t, "spans": line_spans,
                        "page_index": page_index, "indent": 0.0,
                    })
            elif kind == "tocln":
                _, block, t, spans = item
                page_blocks.append({
                    "bbox": block["bbox"], "text": t, "spans": spans,
                    "page_index": page_index, "indent": 0.0,
                })
            else:  # "multi", or "single" when merge_mode is off
                _, block, text, spans = item
                page_blocks.append({
                    "bbox": block["bbox"], "text": text, "spans": spans,
                    "page_index": page_index, "indent": 0.0,
                })
        _flush_singles()
        all_pages.append(page_blocks)
    doc.close()
    return all_pages


def count_blocks(all_pages: list[list[dict]]) -> int:
    return sum(len(p) for p in all_pages)
