import re
import pymupdf

# Translated TOC line: "title . . . . . 43".  The translator emits a FIXED
# 10-dot leader; _format_toc_line recomputes the dot count so the page number
# lands near the right edge of the block, like professional typesetting.
_TOC_LINE_OUT_RE = re.compile(r'^(.*?)\s*(?:\.\s*){4,}\s*([0-9ivxlcdmIVXLCDM]+)$')

# The 14 standard PDF fonts can't encode curly quotes / dashes; they render
# as "?".  Translated text is already ASCII-normalized by the translator, but
# PASSTHROUGH blocks (author lists, name lists) keep the original characters.
_FONT_SAFE_MAP = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-", "…": "...", " ": " ",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_color(color_int: int) -> tuple[float, float, float]:
    r = ((color_int >> 16) & 0xFF) / 255
    g = ((color_int >>  8) & 0xFF) / 255
    b = ( color_int        & 0xFF) / 255
    return (r, g, b)


def _select_font(spans: list) -> str:
    """Closest standard PDF font based on span flags and font name.
    Standard fonts never need embedding, so output stays portable.
    """
    if not spans:
        return "helv"
    first = spans[0]
    flags = first.get("flags", 0)
    fname = first.get("font", "").lower()

    is_bold   = bool(flags & 16) or "bold"    in fname
    is_italic = bool(flags &  2) or "italic"  in fname or "oblique" in fname
    is_mono   = bool(flags &  8) or any(x in fname for x in
                    ("courier", "mono", "typewriter", "consola"))
    is_serif  = bool(flags &  4) or any(x in fname for x in
                    ("times", "roman", "garamond", "georgia", "palatino",
                     "serif", "minion", "bookman", "caslon", "baskerville",
                     "didot", "cambria", "constantia", "charter"))

    if is_mono:
        if is_bold and is_italic: return "cobi"
        if is_bold:               return "cobo"
        if is_italic:             return "coit"
        return "cour"
    if is_serif:
        if is_bold and is_italic: return "tibi"
        if is_bold:               return "tibo"
        if is_italic:             return "tiit"
        return "tiro"
    if is_bold and is_italic:     return "hebi"
    if is_bold:                   return "hebo"
    if is_italic:                 return "heit"
    return "helv"


def _detect_align(block: dict, page_width: float, text: str,
                  col_right: float | None = None) -> int:
    """Infer alignment. Justify only for wide paragraph blocks with enough words."""
    rect = pymupdf.Rect(block["bbox"])

    # Centered: narrow block whose centre ≈ page centre
    block_cx = (rect.x0 + rect.x1) / 2
    if abs(block_cx - page_width / 2) < 15 and rect.width < page_width * 0.65:
        return pymupdf.TEXT_ALIGN_CENTER

    # Right-aligned: block hugs the column's right edge and starts well past
    # the page centre ("Preface" chapter titles, page-number footers).
    if (col_right is not None and abs(rect.x1 - col_right) < 6
            and rect.x0 > page_width * 0.5):
        return pymupdf.TEXT_ALIGN_RIGHT

    # Justify only when block is wide AND text has multiple words
    word_count = len(text.split())
    if rect.width > page_width * 0.50 and word_count >= 6:
        return pymupdf.TEXT_ALIGN_JUSTIFY

    return pymupdf.TEXT_ALIGN_LEFT


def _compute_x_limits(rects: list, page_rect: pymupdf.Rect,
                      obstacles: list) -> list[tuple[float, float]]:
    """For each rect, the horizontal range (min_x, max_x) it may grow into
    without hitting a vertically-overlapping neighbour.  Spanish headings are
    longer than the English original; growing the box sideways into free
    column space preserves the fontsize where growing downward cannot
    (a paragraph usually sits right below a heading).
    """
    if rects:
        col_left  = min(r.x0 for r in rects)
        col_right = max(r.x1 for r in rects)
    else:
        col_left, col_right = page_rect.x0 + 36, page_rect.x1 - 36
    side_margin = 2.0
    pool = list(enumerate(rects)) + [(-1, o) for o in obstacles]
    limits = []
    for i, r in enumerate(rects):
        left, right = col_left, col_right
        for j, other in pool:
            if i == j:
                continue
            y_overlap = min(r.y1, other.y1) - max(r.y0, other.y0)
            if y_overlap <= 1.0:
                continue
            if other.x1 <= r.x0 + 0.5:          # neighbour on the left
                left = max(left, other.x1 + side_margin)
            elif other.x0 >= r.x1 - 0.5:        # neighbour on the right
                right = min(right, other.x0 - side_margin)
        limits.append((min(left, r.x0), max(right, r.x1)))
    return limits


def _visual_obstacles(page) -> list[pymupdf.Rect]:
    """Bboxes of images and 2-D vector drawings (figure boxes, diagrams).
    Text blocks must not extend over them: figures often contain their labels
    as pixels/vectors, so no TEXT block marks the area as occupied and
    _compute_max_y would happily let a paragraph grow over the figure.

    Thin horizontal rules (height ≤ 8 pt) are ignored — capping extension at
    every separator line would reintroduce the font shrinking that the bbox
    extension exists to avoid.
    """
    obstacles: list[pymupdf.Rect] = []
    seen: set[int] = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            for r in page.get_image_rects(xref):
                if r.width > 8 and r.height > 8:
                    obstacles.append(pymupdf.Rect(r))
        except Exception:
            continue
    try:
        for d in page.get_drawings():
            r = pymupdf.Rect(d["rect"])
            if r.width > 8 and r.height > 8:
                obstacles.append(r)
    except Exception:
        pass
    return obstacles


def _compute_max_y(rects: list, page_rect: pymupdf.Rect,
                   obstacles: list | None = None) -> list[float]:
    """For each rect, the lowest y to which it may extend without hitting any
    block in the same column.  Two blocks are considered "same column" when
    their horizontal extents overlap by more than 30 % of the narrower one —
    enough to ignore marginalia and the page number footer.

    `obstacles` (image/drawing bboxes) limit extension the same way text
    blocks do, but don't receive a max_y of their own.

    bottom_margin is small (2 pt) — leaves only a hair between blocks, which
    matters because Spanish text is ~20 % longer than English so headings and
    paragraphs need every extra pixel to avoid being shrunk.
    """
    bottom_margin = 2.0
    page_bottom   = page_rect.y1 - 5.0
    others_pool = list(enumerate(rects)) + [(-1, o) for o in (obstacles or [])]
    max_y = []
    for i, r in enumerate(rects):
        best = page_bottom
        for j, other in others_pool:
            if i == j:
                continue
            if other.y0 <= r.y1 + 0.5:
                continue  # other is above or overlapping vertically
            x_overlap = min(r.x1, other.x1) - max(r.x0, other.x0)
            min_w     = min(r.width, other.width)
            if min_w <= 0 or x_overlap < min_w * 0.30:
                continue  # different column / different layout area
            limit = other.y0 - bottom_margin
            if limit < best:
                best = limit
        max_y.append(max(best, r.y1))
    return max_y


def _candidate_rects(rect, align, max_y, x_limits) -> list:
    """Rects to try, smallest disruption first: original → width-extended
    (direction preserves visual alignment) → height-extended (capped) →
    both.  The downward cap keeps a high-fontsize cover block from
    swallowing half the page when the model returns garbage.
    """
    MAX_EXTEND_RATIO = 2.0
    MAX_EXTEND_ABS   = 40.0

    capped_y = min(
        max_y,
        rect.y0 + rect.height * MAX_EXTEND_RATIO,
        rect.y1 + MAX_EXTEND_ABS,
    )
    min_x, max_x = x_limits

    if align == pymupdf.TEXT_ALIGN_RIGHT:
        wide = pymupdf.Rect(min_x, rect.y0, rect.x1, rect.y1)
    elif align == pymupdf.TEXT_ALIGN_CENTER:
        room = max(0.0, min(rect.x0 - min_x, max_x - rect.x1))
        wide = pymupdf.Rect(rect.x0 - room, rect.y0,
                            rect.x1 + room, rect.y1)
    else:
        wide = pymupdf.Rect(rect.x0, rect.y0, max_x, rect.y1)

    candidates = [rect]
    if wide.width > rect.width + 2:
        candidates.append(wide)
    tall = pymupdf.Rect(rect.x0, rect.y0, rect.x1, capped_y)
    if tall.height > rect.height + 1:
        candidates.append(tall)
    if wide.width > rect.width + 2 and capped_y > rect.y1 + 1:
        candidates.append(pymupdf.Rect(wide.x0, rect.y0, wide.x1, capped_y))
    return candidates


def _base_lineheight(fontname: str) -> float:
    try:
        f = pymupdf.Font(fontname)
        return f.ascender - f.descender
    except Exception:
        return 1.2


def _plan_fit(scratch, rect, text, fontsize, fontname, align, max_y,
              x_limits) -> tuple[float, float | None, pymupdf.Rect]:
    """Find (fontsize, lineheight, rect) that fits, via trial inserts on a
    SCRATCH page (identical metrics to the real insert, zero side effects).

    Cascade: candidate rects at the requested size → tighter leading (a 5–10%
    tighter line spacing is far less visible than a smaller font) → geometric
    fontsize shrink (8 % per try) with the tightest leading.
    """
    candidates = _candidate_rects(rect, align, max_y, x_limits)
    base_lh = _base_lineheight(fontname)

    for cand in candidates:
        if scratch.insert_textbox(
            cand, text, fontsize=fontsize, fontname=fontname, align=align,
        ) >= 0:
            return fontsize, None, cand

    big = candidates[-1]
    for lh_mult in (0.95, 0.90):
        if scratch.insert_textbox(
            big, text, fontsize=fontsize, fontname=fontname, align=align,
            lineheight=base_lh * lh_mult,
        ) >= 0:
            return fontsize, base_lh * lh_mult, big

    fs = fontsize
    for _ in range(16):
        fs = max(5.0, fs * 0.92)
        if scratch.insert_textbox(
            big, text, fontsize=fs, fontname=fontname, align=align,
            lineheight=base_lh * 0.90,
        ) >= 0:
            return fs, base_lh * 0.90, big
        if fs <= 5.0:
            break
    return 5.0, base_lh * 0.90, big


def _format_toc_line(text: str, rect: pymupdf.Rect, fontname: str,
                     fontsize: float) -> str:
    """Recompute a TOC line's dot leader so the page number lands near the
    right edge of the block instead of wherever the fixed 10-dot leader of
    the translator ends.  Falls back to the original text when metrics are
    unavailable or the title leaves no room for a leader.
    """
    m = _TOC_LINE_OUT_RE.match(text.strip())
    if not m:
        return text
    title = m.group(1).rstrip(' .')
    num   = m.group(2)
    try:
        font = pymupdf.Font(fontname)
        tw = font.text_length(title + ' ', fontsize)
        nw = font.text_length(num, fontsize)
        dw = font.text_length('. ', fontsize)
    except Exception:
        return text
    avail = rect.width - tw - nw - fontsize * 1.5  # safety padding
    if avail < dw * 2:
        return f'{title} {num}'
    ndots = min(80, int(avail // dw))
    return f'{title} ' + '. ' * ndots + num


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_translated_pdf(
    pdf_path: str,
    all_pages: list[list[dict]],
    translated_pages: list[list[str]],
    output_path: str,
) -> None:
    doc = pymupdf.open(pdf_path)

    for page_index, (page_blocks, page_texts) in enumerate(
        zip(all_pages, translated_pages)
    ):
        if not page_blocks:
            continue

        page       = doc[page_index]
        page_width = page.rect.width

        # Pre-build Rects (reused in both passes)
        rects = [pymupdf.Rect(b["bbox"]) for b in page_blocks]
        obstacles = _visual_obstacles(page)
        max_y_list = _compute_max_y(rects, page.rect, obstacles)
        x_limits_list = _compute_x_limits(rects, page.rect, obstacles)
        col_right = max(r.x1 for r in rects)

        # Pass 1: redact all text areas at once (original bboxes only, so we
        # don't erase any extended area that belongs to a neighbouring block).
        # PDF_REDACT_IMAGE_NONE preserves images that overlap text.
        for rect in rects:
            page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        # Pass 2a: PLAN every block's fitting size on a scratch page with
        # identical geometry.  Dense dialogue pages otherwise end up a mosaic
        # of per-block shrink levels; knowing all sizes upfront lets same-
        # style blocks be set to one uniform size (next step).
        scratch_doc = pymupdf.open()
        scratch = scratch_doc.new_page(width=page.rect.width,
                                       height=page.rect.height)
        metas: list = []
        plans: list = []
        for block, rect, max_y, x_limits, text in zip(
                page_blocks, rects, max_y_list, x_limits_list, page_texts):
            # Safety net against data loss: the area was already redacted in
            # pass 1, so an empty translation would leave a blank box.  Restore
            # the original text instead — never erase content we can't replace.
            if not text or not text.strip():
                text = block.get("text", "")
            if not text or not text.strip():
                metas.append(None)
                plans.append(None)
                continue
            text = text.translate(_FONT_SAFE_MAP)

            spans      = block.get("spans", [])
            first_span = spans[0] if spans else {}
            fontname   = _select_font(spans)
            fontsize   = first_span.get("size", 11)
            color      = _unpack_color(first_span.get("color", 0))
            align      = _detect_align(block, page_width, text, col_right)
            text       = _format_toc_line(text, rect, fontname, fontsize)

            # Reproduce the paragraph's first-line indent (merged multi-line
            # paragraphs carry it in points) with leading spaces — only for
            # flush-left / justified bodies, never centered or right titles.
            indent_pts = block.get("indent", 0.0)
            if indent_pts > 2 and align in (pymupdf.TEXT_ALIGN_LEFT,
                                            pymupdf.TEXT_ALIGN_JUSTIFY):
                try:
                    sw = pymupdf.Font(fontname).text_length(" ", fontsize)
                    n_spaces = max(1, round(indent_pts / sw)) if sw else 0
                except Exception:
                    n_spaces = 0
                if n_spaces:
                    text = " " * n_spaces + text

            metas.append((text, fontname, fontsize, color, align,
                          rect, max_y, x_limits))
            plans.append(_plan_fit(scratch, rect, text, fontsize, fontname,
                                   align, max_y, x_limits))

        # Pass 2b: per original-size class (≥3 blocks = body text, not a
        # heading), unify to the 25th-PERCENTILE planned size of the class,
        # floored at 80 % of the original.  A page whose paragraphs are all
        # 9.9 pt looks typeset; a mix of 10.8/9.9/9.1/8.4 does not.  The
        # percentile (NOT the minimum) keeps one pathological block from
        # dragging the whole page to the floor — outliers below the chosen
        # size simply keep their own smaller size.
        classes: dict[float, list[int]] = {}
        for idx, meta in enumerate(metas):
            if meta is not None:
                classes.setdefault(round(meta[2], 1), []).append(idx)
        for orig_fs, idxs in classes.items():
            if len(idxs) < 3:
                continue
            planned = sorted(plans[i][0] for i in idxs)
            p25 = planned[int(len(planned) * 0.25)]
            uniform = max(p25, orig_fs * 0.80)
            if uniform >= orig_fs - 0.05:
                continue  # everything already fits at the original size
            for i in idxs:
                if plans[i][0] > uniform + 0.05:
                    text, fontname, _fs, _c, align, rect, max_y, xl = metas[i]
                    plans[i] = _plan_fit(scratch, rect, text, uniform,
                                         fontname, align, max_y, xl)
        scratch_doc.close()

        # Pass 2c: real insertion with the final plan.
        for meta, plan in zip(metas, plans):
            if meta is None:
                continue
            text, fontname, _fs, color, align, rect, _my, _xl = meta
            fs, lh, use_rect = plan
            kwargs = dict(fontsize=fs, fontname=fontname, color=color,
                          align=align)
            if lh is not None:
                kwargs["lineheight"] = lh
            if page.insert_textbox(use_rect, text, **kwargs) < 0:
                # Identical metrics to the scratch plan — should not happen.
                page.insert_textbox(use_rect, text, fontsize=5.0,
                                    fontname=fontname, color=color,
                                    align=align)

    doc.save(output_path)
    doc.close()
