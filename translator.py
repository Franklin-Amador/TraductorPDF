import os
import re
import shutil
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

import glossary

# Sentence boundary: split after . ! ? followed by whitespace + capital letter.
# Negative lookbehind (?<![0-9]) prevents splitting "1. Title" at the list number.
# (?<!\s[A-Z]\.) prevents splitting after single-letter initials: "Edgar F.
# Codd" must stay one sentence — the isolated "Edgar F." fragment otherwise
# makes the model emit the author name twice (bibliography garbage).
_SENT_RE = re.compile(r'(?<![0-9])(?<!\s[A-Z]\.)(?<=[.!?])\s+(?=[A-Z"\(\[])')

# Abbreviations whose "." must not split a sentence ("e.g. the database").
# The dots are temporarily swapped for \x00 around _SENT_RE.split.
_ABBR_PROTECT_RE = re.compile(r'\b(e\.g|i\.e|cf|vs|et al|Fig|fig|Eq|No)\.')

SUPPORTED_PAIRS = {
    ("en", "es"): "Helsinki-NLP/opus-mt-tc-big-en-es",  # big model, best quality
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",          # standard (no tc-big available)
}

LANGUAGES = [
    {"code": "es", "name": "Spanish"},
    {"code": "en", "name": "English"},
]

# ---------------------------------------------------------------------------
# Catálogo de modelos — el usuario elige según la gama de su PC (ver hardware.py)
# ---------------------------------------------------------------------------
# backend "opus": OPUS-MT (Helsinki), un modelo por par, SentencePiece, rápido.
# backend "nllb": NLLB-200 (Meta), un modelo multilingüe, tokenizer HF; más
#                 inteligente (mejor vocabulario técnico, casi sin bucles).
MODELS = {
    "opus": {
        "id": "opus",
        "backend": "opus",
        "label": "OPUS-MT — rápido y ligero",
        "size": "~0.3 GB", "min_ram_gb": 2, "quality_rank": 1,
        "blurb": "El más rápido. Buena calidad. Ideal para equipos básicos.",
    },
    "nllb-600M": {
        "id": "nllb-600M",
        "backend": "nllb",
        "repo": "facebook/nllb-200-distilled-600M",
        "label": "NLLB-200 600M — equilibrado",
        "size": "~1.2 GB", "min_ram_gb": 4, "quality_rank": 2,
        "blurb": "Más inteligente que OPUS, casi igual de rápido. Recomendado.",
    },
    "nllb-1.3B": {
        "id": "nllb-1.3B",
        "backend": "nllb",
        "repo": "facebook/nllb-200-distilled-1.3B",
        "label": "NLLB-200 1.3B — alta calidad",
        "size": "~2.6 GB", "min_ram_gb": 8, "quality_rank": 3,
        "blurb": "Calidad notablemente superior. Más lento en CPU. 8 GB+ RAM.",
    },
    "nllb-3.3B": {
        "id": "nllb-3.3B",
        "backend": "nllb",
        "repo": "facebook/nllb-200-3.3B",
        "label": "NLLB-200 3.3B — máxima calidad",
        "size": "~6.6 GB", "min_ram_gb": 16, "quality_rank": 4,
        "blurb": "Lo mejor sin GPU. Lento en CPU. Para equipos de 16 GB+.",
    },
}

# Códigos FLORES-200 que NLLB exige como token de idioma.
_NLLB_LANG = {"en": "eng_Latn", "es": "spa_Latn"}

DEFAULT_MODEL = "opus"

BATCH_SIZE = 32   # tc-big models are larger; smaller batches avoid OOM
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_ct2")

# Texts matching this pattern carry no translatable content
_TRIVIAL_RE = re.compile(r"^[\d\s.,;\-–—:!?()\[\]{}'\"*#@%^&+=|\\/<>~`©®™°•·…]+$")

# Roman numerals used as page numbers (e.g. "ix", "xvii").  Both all-lowercase
# and all-uppercase forms are accepted.  Length cap of 6 avoids matching short
# real words like "mix" that happen to use Roman-numeral letters.
_ROMAN_RE = re.compile(r'^(?:[ivxlcdm]+|[IVXLCDM]+)$')

# Unicode → ASCII normalization applied BEFORE tokenization.
# Fancy quotes / dashes that are rare in OPUS-MT training data confuse the
# decoder and trigger hallucination loops.
_UNICODE_MAP = str.maketrans({
    "“": '"',   # " left double quotation mark
    "”": '"',   # " right double quotation mark
    "‘": "'",   # ' left single quotation mark
    "’": "'",   # ' right single quotation mark / apostrophe
    "‚": ",",   # ‚ single low-9 quotation mark
    "„": '"',   # „ double low-9 quotation mark
    "–": "-",   # – en dash
    "—": " - ", # — em dash
    "…": "...", # … ellipsis
    " ": " ",   # non-breaking space
    "­": "",    # soft hyphen (invisible, causes tokenizer noise)
    "•": "*",   # • bullet
    "·": "*",   # · middle dot
    "−": "-",   # − minus sign
    "‐": "-",   # ‐ hyphen
    "‑": "-",   # ‑ non-breaking hyphen
    "―": "-",   # ― horizontal bar
})

# Letters with diacritics that are NOT Spanish (German/Nordic/Turkish names:
# "Özcan", "Hölzle").  The OPUS es vocab cannot emit them → UNK ("⁇zcan").
# ASCII-folding the SOURCE gives "Ozcan" — imperfect but readable.  ONLY for
# OPUS: NLLB handles Unicode natively, so folding there would needlessly strip
# accents from names.  Spanish letters (áéíóúüñ) are never touched.
_DIACRITIC_MAP = str.maketrans({
    "ö": "o", "Ö": "O", "ä": "a", "Ä": "A", "å": "a", "Å": "A",
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ß": "ss", "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S",
    "ı": "i", "İ": "I", "ë": "e", "Ë": "E", "ï": "i", "Ï": "I",
    "ã": "a", "Ã": "A", "õ": "o", "Õ": "O", "â": "a", "Â": "A",
    "ê": "e", "Ê": "E", "î": "i", "Î": "I", "ô": "o", "Ô": "O",
    "û": "u", "Û": "U", "à": "a", "À": "A", "è": "e", "È": "E",
    "ì": "i", "Ì": "I", "ò": "o", "Ò": "O", "ù": "u", "Ù": "U",
})

# ---------------------------------------------------------------------------
# Passthrough heuristics — blocks that should NOT be fed to the model
# ---------------------------------------------------------------------------

# Common English words that appear Title-Cased in headings but are NOT proper names.
# Used to distinguish "Martin Kleppmann" (name) from "The Big Ideas" (title text).
_TITLE_COMMON = frozenset({
    'The', 'A', 'An', 'In', 'Of', 'For', 'And', 'Or', 'But', 'To',
    'With', 'At', 'By', 'From', 'Is', 'Are', 'Was', 'Were', 'Be',
    'It', 'As', 'On', 'Up', 'No', 'New', 'All', 'Its', 'This', 'That',
    'Has', 'Had', 'Have', 'Not', 'Can', 'Do', 'Did', 'If', 'We', 'You',
})

# English morphological suffixes: words ending in these are common English
# nouns/adjectives, NOT surnames.  "Tolerance" ends in "ance" → not a surname.
_ENG_SUFFIX_RE = re.compile(
    r'(?:tion|ity|ness|ment|ance|ence|ing|ism|ive|ous|ful|less|able|ible|'
    r'ics|ary|ery|ory|ship|ward|hood|dom|ate|ize|ise|ify|age|ure|ance)$',
    re.IGNORECASE,
)

# Common English content words that may appear as the second word of a 2-word
# heading but are NOT surnames.  Complement to _ENG_SUFFIX_RE for words whose
# endings don't match the suffix pattern (e.g. "Source", "Services").
_COMMON_SECOND_WORDS = frozenset({
    'Source', 'Service', 'Services', 'System', 'Systems', 'Model', 'Models',
    'Design', 'Method', 'Methods', 'Theory', 'Practice', 'Concept', 'Concepts',
    'Pattern', 'Patterns', 'Network', 'Networks', 'Access', 'Control',
    'Storage', 'Memory', 'Cache', 'Index', 'Query', 'Engine', 'Layer',
    'Module', 'Server', 'Client', 'Protocol', 'Format', 'Schema', 'Language',
    'Cloud', 'Stack', 'Platform', 'Framework', 'Version', 'Edition', 'Update',
    'Release', 'Table', 'Graph', 'Tree', 'Queue', 'Store', 'Stream', 'Batch',
    'Cluster', 'Shard', 'Node', 'Replica', 'Leader', 'Follower', 'Peer',
    'Record', 'Schema', 'Object', 'Field', 'Column', 'Row', 'Key', 'Value',
    'Function', 'Class', 'Type', 'Event', 'Error', 'State', 'Context',
})

# TOC entry: text + 4+ dot-space repetitions (or 8+ spaces) + page number.
# Roman numerals (xvii) and Arabic numerals (43) both accepted.
_TOC_SUFFIX_RE = re.compile(
    r'^(.+?)\s*((?:\.\s*){4,}|\s{8,})([\divxlcdmIVXLCDM]+)\s*$'
)

# Adjacent duplicate words (≥3 chars, case-insensitive): "cuadro cuadro" → "cuadro"
_DUP_WORD_RE = re.compile(r'\b(\w{3,})\s+\1\b', re.IGNORECASE)

# Title-Case hyphenated words → space-separated before tokenizing.
# "Data-Intensive" → "Data Intensive", "Trade-Offs" → "Trade Offs".
# Only fires when a Title-Case word (Uppercase + lowercase+) precedes the hyphen,
# so "co-author" and "anti-virus" (lowercase first letter) are left untouched.
_TITLE_HYPHEN_RE = re.compile(r'([A-Z][a-z]+)-([A-Za-z])')

# Leading chapter/section number in TOC titles ("1. ", "2.3. ").
# Stripped before translation and restored afterwards so the model doesn't see
# the number and generate list-like garbage ("1) a 1 (1) del 1.1 1. ...").
_CHAPTER_PREFIX_RE = re.compile(r'^(\d+(?:\.\d+)*\.\s+)')

# Running-header / footer blocks: "Title | 3" (odd pages) or "x | Title"
# (even pages).  The title is translated like a TOC heading; the page-number
# side is preserved verbatim.
_FOOTER_NUM_RIGHT_RE = re.compile(r'^(.+?)\s*\|\s*([0-9ivxlcdmIVXLCDM]+)$')
_FOOTER_NUM_LEFT_RE  = re.compile(r'^([0-9ivxlcdmIVXLCDM]+)\s*\|\s*(.+)$')


def _is_page_num(s: str) -> bool:
    return s.isdigit() or (len(s) <= 6 and _ROMAN_RE.match(s) is not None)


# Bibliography entry prefix "[ 5 ] " — stripped before translation and
# restored afterwards.  The model copies the bracket erratically ("[5][ 5 ]")
# and it pollutes the sentence start.
_REF_PREFIX_RE = re.compile(r'^(\[\s*\d+\s*\]\s*)')

# Cross-references to other sections: (see "Title" on page 12).  The model
# mangles the quoted English title into EN/ES soup.  The whole reference is
# masked with a numeric token — digits survive OPUS-MT verbatim — and
# rendered deterministically afterwards, using the glossary for the title
# when available.
_XREF_PAREN_RE = re.compile(
    r'\(\s*see\s+["“]([^"”]{3,90})["”]\s+on\s+page\s+(\d+)\s*\)',
    re.IGNORECASE)
_XREF_BARE_RE = re.compile(
    r'\bsee\s+["“]([^"”]{3,90})["”]\s+on\s+page\s+(\d+)', re.IGNORECASE)


def _mask_cross_refs(text: str):
    """Replace cross-references with numeric markers.  Returns the masked
    text and a list of (marker, rendered_spanish) to restore after decode."""
    restores: list[tuple[str, str]] = []
    counter = [900]

    def _render(title: str, page: str, parens: bool) -> str:
        es = glossary.lookup(title)
        shown = es if es is not None else title
        body = f'ver "{shown}" en la página {page}'
        return f'({body})' if parens else body

    def _sub(parens: bool):
        def inner(m):
            counter[0] += 1
            marker = f'({counter[0]})'
            restores.append((marker, _render(m.group(1), m.group(2), parens)))
            return marker
        return inner

    text = _XREF_PAREN_RE.sub(_sub(True), text)
    text = _XREF_BARE_RE.sub(_sub(False), text)
    return text, restores


def _split_footer(text: str):
    """Detect a running header/footer.  Returns (prefix, title, suffix) with
    the page-number half folded into prefix/suffix, or None."""
    m = _FOOTER_NUM_RIGHT_RE.match(text)
    if m and _is_page_num(m.group(2)):
        return '', m.group(1).strip(), f' | {m.group(2)}'
    m = _FOOTER_NUM_LEFT_RE.match(text)
    if m and _is_page_num(m.group(1)):
        return f'{m.group(1)} | ', m.group(2).strip(), ''
    return None


def _is_proper_name(text: str) -> bool:
    """True when text is EXACTLY 2 alphabetic Title-Case words that look like a
    person's name.  We require the last word (likely a surname) to be ≥ 5 chars
    and NOT end with a typical English morphological suffix (which would indicate
    a common noun or adjective, not a surname).

    Examples:
        "Martin Kleppmann" → True   (surname: 9 chars, no suffix)
        "Chris Riccomini"  → True   (surname: 9 chars, no suffix)
        "Jay Kreps"        → True   (surname: 5 chars, no suffix)
        "Fault Tolerance"  → False  (Tolerance ends in -ance)
        "The Big"          → False  ('The' is a common word)
        "Chapter 1"        → False  ('1' is not alpha)
    """
    words = text.strip().split()
    if len(words) != 2:
        return False
    first, last = words
    if not (first.isalpha() and last.isalpha()):
        return False
    if not (first[0].isupper() and last[0].isupper()):
        return False
    if first in _TITLE_COMMON or last in _TITLE_COMMON:
        return False
    if len(last) < 5:
        return False
    if _ENG_SUFFIX_RE.search(last):
        return False
    if last in _COMMON_SECOND_WORDS:
        return False
    return True


def _is_name_list(text: str) -> bool:
    """True when text looks like an acknowledgments-style comma-separated list
    of person names.  Heuristic: 5+ commas, >70% Title-Case alpha words, and
    few lowercase content words (which would indicate a normal sentence).
    """
    if text.count(',') < 5:
        return False
    words = [w.strip('.,;:()[]') for w in text.split()]
    alpha = [w for w in words if w.isalpha() and len(w) >= 2]
    if len(alpha) < 8:
        return False
    # Lowercase content words (≥4 chars) indicate sentence structure, not a name list
    lower_content = [w for w in alpha if len(w) >= 4 and w[0].islower()]
    if len(lower_content) > 3:
        return False
    title_ratio = sum(1 for w in alpha if w[0].isupper()) / len(alpha)
    return title_ratio > 0.70


_INITIAL_RE = re.compile(r'^[A-Z]\.?$')

# Lowercase particles that legitimately appear inside author names/lists
_NAME_PARTICLES = frozenset({
    'and', 'y', 'van', 'von', 'de', 'der', 'den', 'del', 'da', 'dos',
    'la', 'le', 'el', 'di', 'du', 'al', 'bin', 'ter',
})


def _is_author_sentence(text: str) -> bool:
    """True when a sentence is a bibliography author list: nearly all words
    are Title-Case names, initials ("F.") or name particles ("van", "and").
    Only called for sentences inside "[ N ]"-prefixed reference blocks, so
    2-word headings can't be misclassified.  Author names must not be
    translated — the model loops on them ("Pedro Rui de Pedro Ruy Pedro…").
    """
    words = text.rstrip('.').split()
    if not 2 <= len(words) <= 30:
        return False
    total = 0
    name_like = 0
    for w in words:
        core = w.strip('.,;:()[]"\'')
        if not core:
            continue
        total += 1
        if (_INITIAL_RE.match(core)
                or core.lower() in _NAME_PARTICLES
                or (core[0].isupper() and core.replace('-', '').isalpha())):
            name_like += 1
    return total >= 2 and name_like == total


def _split_toc(text: str):
    """Detect a table-of-contents entry.

    Returns (title, suffix, should_translate) when text matches "title . . . . N":
      • title            – the heading text to (possibly) translate
      • suffix           – normalized dot-leader + page number to re-append
      • should_translate – False for single-word titles (Preface, Index…) which
                           the model handles poorly; the English word is kept as-is.

    Returns None when the text is NOT a TOC entry.
    """
    m = _TOC_SUFFIX_RE.match(text.strip())
    if not m:
        return None
    title = m.group(1).rstrip('. ')
    page  = m.group(3)
    suffix = f" . . . . . . . . . . {page}"
    translate = len(title.split()) >= 2
    return title, suffix, translate


# ---------------------------------------------------------------------------
# Core translation helpers
# ---------------------------------------------------------------------------

def get_languages() -> list[dict]:
    return LANGUAGES


def _normalize(text: str) -> str:
    """Replace Unicode punctuation with ASCII equivalents, then de-hyphenate
    Title-Case compound words so the tokenizer handles them as separate tokens.
    """
    text = text.translate(_UNICODE_MAP)
    text = _TITLE_HYPHEN_RE.sub(r'\1 \2', text)  # "Data-Intensive" → "Data Intensive"
    return text


def _needs_translation(text: str) -> bool:
    s = text.strip()
    if len(s) <= 1:
        return False
    if _TRIVIAL_RE.match(s):
        return False
    # Roman-numeral page numbers ("ix", "xvii") appear as isolated blocks on
    # TOC pages; sending them to the model causes severe hallucination loops.
    if len(s) <= 6 and _ROMAN_RE.match(s):
        return False
    return True


def _ct2_dir(model_name: str) -> str:
    return os.path.join(_MODELS_DIR, model_name.replace("/", "_"))


def _build_model(model_name: str, out_dir: str, copy_spm: bool = True) -> None:
    conv = ctranslate2.converters.TransformersConverter(model_name)
    conv.convert(out_dir, quantization="int8")
    if not copy_spm:
        return  # NLLB uses the HF tokenizer at runtime, not a copied .spm
    hf_dir = snapshot_download(model_name, local_files_only=True)
    candidates = [
        ("source.spm", "source.spm"),
        ("target.spm", "target.spm"),
        ("sentencepiece.bpe.model", "source.spm"),
    ]
    for src_name, dst_name in candidates:
        src = os.path.join(hf_dir, src_name)
        dst = os.path.join(out_dir, dst_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


class Engine:
    """Wraps a loaded translation model so the rest of the pipeline never sees
    the backend differences.  OPUS-MT and NLLB tokenize and language-tag
    differently; everything downstream (heuristics, layout) stays the same."""

    def __init__(self, backend, ct2, *, src_sp=None, tgt_sp=None,
                 hf_tok=None, src_lang=None, tgt_lang=None, model_id="opus"):
        self.backend = backend
        self.ct2 = ct2
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.hf_tok = hf_tok
        self.src_lang = src_lang   # FLORES code, NLLB only
        self.tgt_lang = tgt_lang
        self.model_id = model_id

    @property
    def fold_diacritics(self) -> bool:
        """OPUS was trained on ASCII; folding "Özcan"→"Ozcan" and splitting
        "Data-Intensive" helps it but HURTS NLLB (which handles Unicode)."""
        return self.backend == "opus"

    def encode(self, text: str) -> list[str]:
        if self.backend == "nllb":
            self.hf_tok.src_lang = self.src_lang
            ids = self.hf_tok(text).input_ids
            return self.hf_tok.convert_ids_to_tokens(ids)
        return self.src_sp.encode(text, out_type=str)

    def translate(self, tokenized, **kw) -> list[str]:
        if self.backend == "nllb":
            kw["target_prefix"] = [[self.tgt_lang]] * len(tokenized)
        results = self.ct2.translate_batch(tokenized, **kw)
        out = []
        for r in results:
            hyp = r.hypotheses[0]
            if self.backend == "nllb":
                if hyp and hyp[0] == self.tgt_lang:
                    hyp = hyp[1:]
                out.append(self.hf_tok.decode(
                    self.hf_tok.convert_tokens_to_ids(hyp),
                    skip_special_tokens=True))
            else:
                out.append(self.tgt_sp.decode(hyp))
        return out


def model_exists_locally(model_id: str, source: str, target: str) -> bool:
    """True if the chosen model is already downloaded+converted (no network
    needed).  Lets the GUI warn before a multi-GB download."""
    spec = MODELS.get(model_id, MODELS[DEFAULT_MODEL])
    if spec["backend"] == "opus":
        repo = SUPPORTED_PAIRS.get((source, target))
        return repo is not None and os.path.isdir(_ct2_dir(repo))
    return os.path.isdir(_ct2_dir(spec["repo"]))


def load_engine(source: str, target: str, model_id: str = DEFAULT_MODEL,
                status_callback=None) -> Engine:
    spec = MODELS.get(model_id, MODELS[DEFAULT_MODEL])
    n_threads = max(1, os.cpu_count() or 1)

    def _say(msg):
        if status_callback:
            status_callback(msg)

    if spec["backend"] == "opus":
        key = (source, target)
        if key not in SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported language pair: {source} -> {target}")
        repo = SUPPORTED_PAIRS[key]
        out_dir = _ct2_dir(repo)
        if not os.path.isdir(out_dir):
            os.makedirs(_MODELS_DIR, exist_ok=True)
            _say(f"Descargando y optimizando {spec['label']} "
                 f"({spec['size']}, solo la primera vez)…")
            _build_model(repo, out_dir, copy_spm=True)
        ct2 = ctranslate2.Translator(out_dir, device="cpu", inter_threads=1,
                                     intra_threads=n_threads,
                                     compute_type="int8_float32")
        src_sp = spm.SentencePieceProcessor()
        src_sp.load(os.path.join(out_dir, "source.spm"))
        tgt_sp = spm.SentencePieceProcessor()
        tgt_sp.load(os.path.join(out_dir, "target.spm"))
        return Engine("opus", ct2, src_sp=src_sp, tgt_sp=tgt_sp,
                      model_id=model_id)

    # ----- NLLB backend -----
    if source not in _NLLB_LANG or target not in _NLLB_LANG:
        raise ValueError(f"NLLB no soporta el par {source}->{target} aquí.")
    try:
        import transformers
    except ImportError:
        raise RuntimeError(
            "El modelo NLLB necesita la librería 'transformers'.\n"
            "Instálala con:  pip install transformers")
    repo = spec["repo"]
    out_dir = _ct2_dir(repo)
    if not os.path.isdir(out_dir):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        _say(f"Descargando y optimizando {spec['label']} "
             f"({spec['size']}, solo la primera vez)…")
        _build_model(repo, out_dir, copy_spm=False)
    ct2 = ctranslate2.Translator(out_dir, device="cpu", inter_threads=1,
                                 intra_threads=n_threads,
                                 compute_type="int8_float32")
    hf_tok = transformers.AutoTokenizer.from_pretrained(
        repo, src_lang=_NLLB_LANG[source])
    return Engine("nllb", ct2, hf_tok=hf_tok,
                  src_lang=_NLLB_LANG[source], tgt_lang=_NLLB_LANG[target],
                  model_id=model_id)


def load_model(source: str, target: str, status_callback=None):
    """Compat: devuelve un Engine OPUS-MT (firma antigua de 3-tupla obsoleta)."""
    return load_engine(source, target, DEFAULT_MODEL, status_callback)


def _split_sentences(text: str) -> list[str]:
    # Protect abbreviation dots ("e.g.", "vs.") so they don't split sentences;
    # restored before returning (the model must see the real ".").
    protected = _ABBR_PROTECT_RE.sub(
        lambda m: m.group(0).replace('.', '\x00'), text.strip())
    parts = _SENT_RE.split(protected)
    return [p.replace('\x00', '.').strip() for p in parts if p.strip()]


# SentencePiece unknown-token glyph: U+2047 "⁇" (DOUBLE QUESTION MARK).
# The decoder emits this character when a hypothesis token has no vocabulary entry.
_UNK_GLYPH = '⁇'

# The es vocab cannot emit some word-initial accented capitals: the model
# produces "⁇ndices" where it means "Índices".  Blindly deleting the UNK
# leaves a mutilated word ("ndices").  This table maps the headless fragment
# to its missing capital for frequent Spanish words.
_UNK_HEAD_FIXES = {
    'ndice': 'Í', 'ndices': 'Í', 'dolo': 'Í', 'dolos': 'Í',
    'ltimo': 'Ú', 'ltima': 'Ú', 'ltimos': 'Ú', 'ltimas': 'Ú',
    'nico': 'Ú', 'nica': 'Ú', 'nicos': 'Ú', 'nicas': 'Ú',
    'til': 'Ú', 'tiles': 'Ú',
    'xito': 'É', 'xitos': 'É', 'poca': 'É', 'pocas': 'É',
    'tica': 'É', 'ticas': 'É', 'nfasis': 'É',
    'rea': 'Á', 'reas': 'Á', 'frica': 'Á', 'lgebra': 'Á',
    'mbito': 'Á', 'mbitos': 'Á', 'rbol': 'Á', 'rboles': 'Á',
    'cido': 'Á', 'cidos': 'Á', 'ngulo': 'Á', 'ngulos': 'Á',
    'tomo': 'Á', 'tomos': 'Á', 'rabe': 'Á', 'rabes': 'Á', 'lbum': 'Á',
    'ptimo': 'Ó', 'ptima': 'Ó', 'ptimos': 'Ó', 'ptimas': 'Ó',
    'rgano': 'Ó', 'rganos': 'Ó', 'pera': 'Ó', 'peras': 'Ó',
    'ptica': 'Ó', 'xido': 'Ó', 'xidos': 'Ó',
}

_UNK_HEAD_RE = re.compile(r'⁇\s?([a-záéíóúüñ]{3,})')


def _recover_unk_heads(text: str) -> str:
    def _fix(m):
        frag = m.group(1)
        cap = _UNK_HEAD_FIXES.get(frag.lower())
        return cap + frag if cap else m.group(0)
    return _UNK_HEAD_RE.sub(_fix, text)

# Continuation patterns — signal the model ran past EOS and started a second
# translation.  The middle group skips UNK glyphs and whitespace:
#   • ⁇ "⁇" = SentencePiece UNK token emitted between sentences
#   • ¿ / ¡      = Spanish opening punctuation that starts the next sentence
# Lookahead includes digits so "Edición. 2a edición de…" is cut at "Edición."
# (the "2a" is a paraphrase restart, not real content).
# (?<!\s[A-Z]\.) keeps author initials intact: without it, the OUTPUT
# "Edgar F. Codd, …" is cut to "Edgar F." (same root cause as in _SENT_RE).
_CONT_RE = re.compile(r'(?<!\s[A-Z]\.)(?<=[.!?])[\s⁇]*(?=[A-ZÁÉÍÓÚÜÑ¿¡0-9])')
_CONT_LOWER_RE = re.compile(r'(?<=[.!?])[\s⁇¿¡]+(?=[a-záéíóúüñ]{2,})')
_DOT_DOT_RE    = re.compile(r'(?<=[.!?])[\s⁇]*\.')


def _is_degenerate(src: str, out: str) -> bool:
    """True when the model's output for a SHORT input is clearly broken, so the
    caller should fall back to the glossary or the original.

    OPUS-MT mangles isolated heading words badly:
        "Writing"  -> "Ww w"      (no real word)
        "Coding"   -> ""          (empty)
        "Tutoring" -> "Tutotutoring"
    We only judge short inputs (≤3 words) — long sentences are left alone, as
    their occasional flaws aren't worth the risk of a false positive.
    """
    out = out.strip()
    src_words = src.split()
    if len(src_words) > 3:
        return False
    if not out:
        return True
    # No token carries 3+ alphabetic characters → "Ww w", "w w w", "1 1 1".
    if not any(sum(c.isalpha() for c in tok) >= 3 for tok in out.split()):
        return True
    # A single output word that is the source word glued to a fragment of
    # itself: "Tutoring" -> "Tutotutoring", "Phoenix" -> "Phoephoenix".
    if len(src_words) == 1 and len(out.split()) == 1:
        o = re.sub(r'[^a-záéíóúüñ]', '', out.lower())
        s = re.sub(r'[^a-záéíóúüñ]', '', src.lower())
        if len(s) >= 4 and o.count(s[:4]) >= 2:
            return True
    return False


def _collapse_repeated_ngrams(text: str) -> str:
    """Remove ADJACENT duplicate word sequences: "se detuvo en seco se detuvo
    en seco" → "se detuvo en seco".  The model restarts mid-sentence and
    repeats whole phrases, especially in fiction prose.  Legitimate adjacent
    identical n-grams (n ≥ 2) are practically nonexistent in Spanish; n = 1
    is handled separately by _DUP_WORD_RE with its own safeguards.
    """
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(6, 1, -1):
            i = 0
            while i + 2 * n <= len(words):
                if ([w.lower() for w in words[i:i + n]]
                        == [w.lower() for w in words[i + n:i + 2 * n]]):
                    del words[i + n:i + 2 * n]
                    changed = True
                else:
                    i += 1
    return ' '.join(words)


def _truncate_output(text: str) -> str:
    """Clip at the first sentence boundary that looks like a second translation.

    Also handles:
    - Adjacent duplicate words ("cuadro cuadro" → "cuadro")
    - First-word repetition within 4 positions ("Edición 2a edición..." → "Edición 2a")
    """
    text = text.strip()

    # Recover word-initial accented capitals the decoder emitted as UNK
    # ("⁇ndices" → "Índices") BEFORE deleting the remaining glyphs.
    text = _recover_unk_heads(text)

    # Remove isolated SentencePiece UNK glyphs (U+2047 "⁇")
    text = text.replace(_UNK_GLYPH, ' ')

    candidates = []
    for pat in (_CONT_RE, _CONT_LOWER_RE, _DOT_DOT_RE):
        m = pat.search(text)
        if m:
            candidates.append(m)
    if candidates:
        earliest = min(candidates, key=lambda x: x.start())
        text = text[: earliest.start()].strip()

    # Collapse adjacent duplicate words (case-insensitive, ≥3 chars)
    text = _DUP_WORD_RE.sub(r'\1', text)

    # Collapse adjacent duplicate phrases ("se detuvo en seco se detuvo en
    # seco" → once) — model restarts, frequent in fiction prose
    text = _collapse_repeated_ngrams(text)

    # First-word repetition within 2 positions: "Edición 2a edición" → "Edición 2a"
    # Limited to k ≤ 2 to avoid truncating legitimate phrases like
    # "Capítulo 1 El Capítulo de las compensaciones..." at k=3.
    words = text.split()
    if len(words) >= 3:
        first_lower = words[0].lower().strip('¿¡')
        for k in range(1, min(3, len(words))):
            if words[k].lower().rstrip('.,;:') == first_lower and len(first_lower) >= 4:
                text = ' '.join(words[:k]).rstrip('.,;: ')
                break

    return text.strip()


# Function words that may legitimately repeat inside a heading and must not
# count as "content echo", nor be left dangling after a truncation.
_FUNC_WORDS = frozenset({
    'de', 'del', 'la', 'el', 'lo', 'los', 'las', 'y', 'e', 'o', 'u', 'a',
    'en', 'un', 'una', 'unos', 'unas', 'para', 'con', 'por', 'que', 'al',
    'su', 'sus', 'se', 'es', 'son', 'versus', 'vs', 'frente', 'entre',
    'sobre', 'contra', 'hacia', 'como', 'más', 'menos',
    'the', 'of', 'and', 'in', 'for', 'to', 'on', 'or',
})

_ACCENT_MAP = str.maketrans("áéíóúüàèìòùñ", "aeiouuaeioun")

# Internal sentence boundary inside a heading (not at the very end, not after
# a digit).  A heading is a single phrase: any internal boundary followed by
# more text is a candidate paraphrase restart.
_HEAD_BOUND_RE = re.compile(r'(?<![0-9])(?<=[.!?])\s*(?=\S)')


def _norm_word(w: str) -> str:
    """Lowercase, fold accents, strip non-letters — for echo comparison."""
    return re.sub(r'[^a-zn]', '', w.lower().translate(_ACCENT_MAP))


def _has_dup_prefixes(text: str, min_len: int = 4) -> bool:
    """True when the text repeats a content-word prefix ("Shared-Memory,
    Shared-Disk...").  Echo compaction is skipped for such sources because
    their translations legitimately repeat words."""
    seen = set()
    for w in re.findall(r"[A-Za-zÀ-ÿ]+", text):
        nw = _norm_word(w)
        if len(nw) < min_len or nw in _FUNC_WORDS:
            continue
        p = nw[:5]
        if p in seen:
            return True
        seen.add(p)
    return False


def _compact_heading(text: str, allow_word_echo: bool = True) -> str:
    """Drop alternative phrasings the model appends to heading translations.

    Two cuts, in order:
    1. Internal sentence boundary whose continuation echoes a content word
       from before the boundary → paraphrase restart, cut at the boundary:
       "¿Quién debería leer este libro?, quién deberia leerlo." → "...libro?"
       (A boundary with NO echo is kept: "...O'Reilly Media, Inc. 1005
       Gravenstein Highway" is legitimate.)
    2. A content word (≥5 normalized chars, prefix comparison so
       "Región"/"Regiones" match) reappearing later in the heading → cut at
       the repeat and drop dangling function words:
       "Almacenamiento Orientado Columna de la columna" →
       "Almacenamiento Orientado Columna"
       Only applied to short outputs (≤10 words) of sources without repeated
       content words (allow_word_echo).
    """
    m = _HEAD_BOUND_RE.search(text)
    if m:
        before, after = text[:m.start()], text[m.start():]
        bp = {_norm_word(w)[:5] for w in before.split()
              if len(_norm_word(w)) >= 5 and _norm_word(w) not in _FUNC_WORDS}
        ap = {_norm_word(w)[:5] for w in after.split()
              if len(_norm_word(w)) >= 5 and _norm_word(w) not in _FUNC_WORDS}
        # Cut only when the kept part is a substantial share of the output.
        # Bibliography-style blocks where the model duplicates the author
        # ("Edgar F. [5] Edgar Edgard F. Codd, ...") would otherwise lose the
        # entire reference to a 2-word "before".
        if bp & ap and len(before) >= 0.4 * len(text):
            text = before.rstrip()

    words = text.split()
    if allow_word_echo and 3 <= len(words) <= 10:
        seen: set[str] = set()
        cut = None
        for idx, w in enumerate(words):
            nw = _norm_word(w)
            if len(nw) < 5 or nw in _FUNC_WORDS:
                continue
            p = nw[:5]
            # Only cut repeats in the trailing 40% of the heading: Spanish
            # legitimately repeats the head noun in "X versus Y" headings
            # ("Sistemas operativos versus sistemas analíticos" — repeat at
            # 50% — must survive; "...Columna de la columna" at 83% is echo).
            if p in seen and idx >= 2 and idx / len(words) >= 0.6:
                cut = idx
                break
            seen.add(p)
        if cut is not None:
            kept = words[:cut]
            while kept and (_norm_word(kept[-1]) in _FUNC_WORDS
                            or not _norm_word(kept[-1])):
                kept.pop()
            if kept:
                text = ' '.join(kept)

    return text.rstrip(',;: ')


# Space before punctuation ("parte : la") — decoder artifact in headings
_SPACE_PUNCT_RE = re.compile(r'\s+([:;,.!?%])')


def _polish_heading(text: str) -> str:
    """Cosmetic cleanup for heading/TOC/footer translations: the model often
    lowercases Title-Case lines and inserts a space before punctuation."""
    text = _SPACE_PUNCT_RE.sub(r'\1', text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def translate_batch(
    texts: list[str],
    src_sp: spm.SentencePieceProcessor,
    tgt_sp: spm.SentencePieceProcessor,
    translator: ctranslate2.Translator,
    progress_callback=None,
) -> list[str]:
    total = len(texts)

    # Start with originals so trivial / passthrough slots are already correct
    results = list(texts)

    # Build work list with special-case handling per block
    work_indices:    list[int]       = []
    sentence_groups: list[list[str]] = []
    # For TOC entries: store (prefix, suffix) keyed by work-group index.
    # prefix = chapter number like "1. " (prepended before translated title)
    # suffix = dot-leader + page number (appended after)
    toc_wrappers:   dict[int, tuple[str, str]] = {}
    # Indices where we appended a "sentinel period" to coax OPUS-MT into emitting
    # EOS.  The trailing "." is stripped from the final translation.
    sentinel_added: set[int] = set()
    # Groups whose SOURCE repeats a content word ("Shared-Memory, Shared-Disk
    # and Shared-Nothing...") — word-echo compaction would truncate legitimate
    # repetition, so it is disabled for them.
    echo_unsafe: set[int] = set()
    # Sentinel groups short enough to be real headings (≤12 source words).
    # Only these get _compact_heading: longer sentinel blocks (bibliography
    # references, address blocks) would lose content to a paraphrase cut.
    heading_groups: set[int] = set()
    # Per-group cross-reference restores: [(marker, rendered_es), ...]
    xref_restores: dict[int, list[tuple[str, str]]] = {}
    # Per-group literal prefix re-attached after translation ("[ 5 ] ")
    ref_prefixes: dict[int, str] = {}

    skipped = 0
    for i, t in enumerate(texts):
        s = t.strip()

        if not _needs_translation(t):
            skipped += 1
            continue

        # Whole-block glossary hit: deterministic translation, skip the model.
        # Catches headings the model reliably mangles ("Fault Tolerance",
        # "Data Warehousing") and single-word section titles.
        gl = glossary.lookup(s)
        if gl is not None:
            results[i] = gl
            skipped += 1
            continue

        # Proper names (exactly 2 Title-Case alpha words, unusual surname):
        # keep original to avoid decoder loops on rare tokens.
        # Also check after stripping leading punctuation ("& Chris Riccomini",
        # "—Will Wilson") so attribution-style blocks are caught as well.
        s_core = s.lstrip('—–-&"\'*() —–')
        if _is_proper_name(s) or _is_proper_name(s_core):
            skipped += 1
            continue

        # Comma-separated lists of person names (acknowledgments pages):
        # pass through unchanged — model hallucinates heavily on these.
        if _is_name_list(s):
            skipped += 1
            continue

        # Table-of-contents entry: translate only the title, restore the suffix.
        # We do NOT apply _is_proper_name here — TOC titles are technical terms.
        # Running header/footer ("Operational Versus Analytical Systems | 3",
        # "x | Table of Contents"): translate the title like a TOC heading,
        # keep the page-number side verbatim.
        foot = _split_footer(s)
        if foot is not None:
            f_prefix, f_title, f_suffix = foot
            gl = glossary.lookup(f_title)
            if gl is not None:
                results[i] = f_prefix + gl + f_suffix
                skipped += 1
            elif not _needs_translation(f_title):
                results[i] = f_prefix + f_title + f_suffix
                skipped += 1
            else:
                if _has_dup_prefixes(f_title):
                    echo_unsafe.add(len(work_indices))
                normalized = _normalize(f_title).rstrip('.,;: ') + '.'
                sents = _split_sentences(normalized) or [normalized]
                toc_wrappers[len(work_indices)] = (f_prefix, f_suffix)
                work_indices.append(i)
                sentence_groups.append(sents)
            continue

        toc = _split_toc(s)
        if toc is not None:
            toc_title, toc_suffix, toc_translate = toc
            # Strip leading chapter/section number ("1. ", "2.3. ") so the
            # model doesn't see list-like digits and generate "1) a 1 (1)..."
            # garbage — and so the glossary can match the bare title.
            ch_prefix = ''
            ch_m = _CHAPTER_PREFIX_RE.match(toc_title)
            if ch_m:
                ch_prefix = ch_m.group(1)
                toc_title  = toc_title[ch_m.end():]
            gl = glossary.lookup(toc_title)
            if gl is not None:
                results[i] = ch_prefix + gl + toc_suffix
                skipped += 1
            elif not toc_translate or not _needs_translation(toc_title):
                # Single-word title or trivial: output as-is with clean suffix
                results[i] = ch_prefix + toc_title + toc_suffix
                skipped += 1
            else:
                if _has_dup_prefixes(toc_title):
                    echo_unsafe.add(len(work_indices))
                # Append a sentence-terminator before tokenizing.  OPUS-MT
                # was trained on full sentences and treats an isolated noun
                # phrase as the start of one, generating paraphrases to fill
                # max_decoding_length.  Adding "." makes the model recognize
                # the input as a complete sentence and emit EOS reliably; the
                # trailing "." in the output is then cleaned up after decode.
                normalized = _normalize(toc_title).rstrip('.,;: ') + '.'
                sents = _split_sentences(normalized) or [normalized]
                toc_wrappers[len(work_indices)] = (ch_prefix, toc_suffix)
                work_indices.append(i)
                sentence_groups.append(sents)
            continue

        # Normal text block.  Short blocks lacking a sentence terminator get
        # a sentinel "." appended before tokenization — same trick as the TOC
        # path.  Without it OPUS-MT treats inputs like "Designing Data-Intensive
        # Applications" as the start of a sentence and generates paraphrases
        # ("uso intensivo de datos diseño Data Datos aplicación intensiva…").
        # The trailing "." is stripped from the translation after decode.
        masked, restores = _mask_cross_refs(t)
        if restores:
            xref_restores[len(work_indices)] = restores
        normalized = _normalize(masked).rstrip()
        ref_m = _REF_PREFIX_RE.match(normalized)
        if ref_m:
            ref_prefixes[len(work_indices)] = ref_m.group(1).strip() + ' '
            normalized = normalized[ref_m.end():]
        word_count = len(normalized.split())
        if (word_count <= 30 and normalized
                and normalized[-1] not in '.!?:;'):
            sentinel_added.add(len(work_indices))
            if word_count <= 12:
                heading_groups.add(len(work_indices))
            if _has_dup_prefixes(normalized):
                echo_unsafe.add(len(work_indices))
            normalized = normalized + '.'

        sents = _split_sentences(normalized)
        if not sents:
            sents = [normalized]
        work_indices.append(i)
        sentence_groups.append(sents)

    if not work_indices:
        if progress_callback:
            progress_callback(total, total)
        return results

    # Flatten all sentences and tokenize.  Track which flat indices belong to
    # TOC titles — they share the regular max_tgt schedule but get an even
    # tighter cap because TOC titles are isolated headings the model would
    # otherwise pad with alternative phrasings ("Datos Almacenamiento de datos
    # Almacenar depósito De").
    flat_sentences: list[str] = []
    toc_flat_indices: set[int] = set()
    ref_flat_indices: set[int] = set()
    for group_idx, sents in enumerate(sentence_groups):
        if group_idx in toc_wrappers:
            for sent in sents:
                toc_flat_indices.add(len(flat_sentences))
                flat_sentences.append(sent)
        else:
            if group_idx in ref_prefixes:
                for k in range(len(flat_sentences),
                               len(flat_sentences) + len(sents)):
                    ref_flat_indices.add(k)
            flat_sentences.extend(sents)

    all_tokens = [src_sp.encode(s, out_type=str) for s in flat_sentences]

    # Author-list sentences inside bibliography entries are passed through
    # untranslated (only "and" → "y"): the model loops on name sequences.
    author_flat: set[int] = set()
    for k in ref_flat_indices:
        if _is_author_sentence(flat_sentences[k]):
            author_flat.add(k)

    # Split flat sentences into TOC titles vs the rest — they get processed
    # in independent batches with different decoding parameters.  TOC titles
    # need a tight cap + length_penalty < 1 to stop the model from appending
    # paraphrases ("Datos Almacenamiento de datos Almacenar depósito De").
    toc_pool     = [k for k in range(len(all_tokens)) if k in toc_flat_indices]
    non_toc_pool = [k for k in range(len(all_tokens))
                    if k not in toc_flat_indices and k not in author_flat]

    toc_pool.sort(key=lambda k: len(all_tokens[k]))
    non_toc_pool.sort(key=lambda k: len(all_tokens[k]))

    translated_flat = [""] * len(flat_sentences)
    for k in author_flat:
        translated_flat[k] = re.sub(r'\band\b', 'y', flat_sentences[k])
    done = skipped

    def _make_buckets(pool: list[int]) -> list[list[int]]:
        """Group consecutive items so each bucket has items of similar length.
        Bucket grows while max_len <= min_len * 1.5 AND bucket size < BATCH_SIZE.

        This keeps max_decoding_length tight per batch: a 4-token "2nd Edition"
        no longer shares a batch with a 20-token paragraph and inherits its
        cap of 28 — instead it lands with other ~4-token items and gets cap 8.
        """
        buckets: list[list[int]] = []
        current: list[int] = []
        cur_min = 0
        for idx in pool:
            L = len(all_tokens[idx])
            if not current:
                current = [idx]
                cur_min = L
            elif L <= cur_min * 1.25 and len(current) < BATCH_SIZE:
                current.append(idx)
            else:
                buckets.append(current)
                current = [idx]
                cur_min = L
        if current:
            buckets.append(current)
        return buckets

    def _run(pool: list[int], is_toc: bool) -> None:
        nonlocal done
        for chunk in _make_buckets(pool):
            tokenized = [all_tokens[j] for j in chunk]
            max_src   = max(len(t) for t in tokenized)

            if is_toc:
                # TOC titles have "." appended so the model knows it's a
                # complete sentence.  The model emits EOS naturally; cap of
                # max_src + 5 leaves room for Spanish verbosity without
                # inviting paraphrases.
                max_tgt = max_src + 5
                length_penalty = 1.0
            elif max_src <= 3:
                max_tgt = max_src + 3
                length_penalty = 1.0
            elif max_src <= 5:
                max_tgt = max_src + 4
                length_penalty = 1.0
            elif max_src <= 20:
                max_tgt = max_src + 8
                length_penalty = 1.0
            else:
                max_tgt = min(max_src + 15, 200)
                length_penalty = 1.0

            translated = translator.translate_batch(
                tokenized,
                # beam 4 (era 2): mejor búsqueda de hipótesis = mejor
                # gramática, especialmente en prosa.  ~1.5-2× más lento en
                # CPU; medido sin regresiones con eval.py.
                beam_size=4,
                max_decoding_length=max_tgt,
                length_penalty=length_penalty,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
            )
            for j, result in zip(chunk, translated):
                raw = tgt_sp.decode(result.hypotheses[0])
                translated_flat[j] = _truncate_output(raw)

            done += len(chunk)
            if progress_callback:
                progress_callback(min(done, total), total)

    _run(non_toc_pool, is_toc=False)
    _run(toc_pool,     is_toc=True)

    # Rejoin translated sentences back into their original text slots
    flat_iter = iter(translated_flat)
    for group_idx, sents in enumerate(sentence_groups):
        translated_sents = [next(flat_iter) for _ in sents]
        translation = " ".join(translated_sents)
        # Wrap with TOC prefix + suffix if this was a TOC entry. TOC titles and
        # sentinel headings get a pass through _compact_heading to drop the
        # alternative phrasings the model often appends to short inputs.
        if group_idx in toc_wrappers:
            ch_prefix, toc_suffix = toc_wrappers[group_idx]
            translation = _compact_heading(
                translation, group_idx not in echo_unsafe)
            # Drop trailing "." that came from the sentence-terminator hack
            translation = translation.rstrip('.,;: ')
            translation = ch_prefix + _polish_heading(translation) + toc_suffix
        elif group_idx in sentinel_added:
            if group_idx in heading_groups:
                translation = _compact_heading(
                    translation, group_idx not in echo_unsafe)
                translation = _polish_heading(translation)
            # Drop the sentinel "." we appended pre-tokenization
            translation = translation.rstrip()
            if translation.endswith('.'):
                translation = translation[:-1].rstrip()

        # Re-attach bibliography prefix and render masked cross-references
        if group_idx in ref_prefixes:
            translation = ref_prefixes[group_idx] + translation
        for marker, rendered in xref_restores.get(group_idx, []):
            if marker in translation:
                translation = translation.replace(marker, rendered, 1)
            else:
                # Tolerate spacing the model may add inside the parens
                loose = re.compile(
                    r'\(\s*' + re.escape(marker.strip('()')) + r'\s*\)')
                translation, n = loose.subn(rendered, translation, 1)
                if n == 0:
                    # Marker lost in decoding — append so no info disappears
                    translation = translation.rstrip() + ' ' + rendered

        # Degenerate-output guard: short headings the model wrecked
        # ("Writing" -> "Ww w", "Coding" -> "") fall back to the glossary, then
        # to the original word.  Skipped for TOC entries (their dot-leader
        # suffix would confuse the check).
        if group_idx not in toc_wrappers:
            orig = texts[work_indices[group_idx]]
            if _is_degenerate(orig, translation):
                gl = glossary.lookup(orig.strip())
                translation = gl if gl is not None else orig.strip()

        results[work_indices[group_idx]] = translation

    return results
