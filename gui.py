import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import extractor
import translator as tr
import builder


class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PDF Translator")
        self.root.geometry("720x420")
        self.root.minsize(640, 380)
        self.root.resizable(True, True)

        self.queue: queue.Queue = queue.Queue()
        self._languages: list[dict] = tr.get_languages()

        self._pdf_path_var = tk.StringVar()
        self._source_var = tk.StringVar(value="es")
        self._target_var = tk.StringVar(value="en")
        self._output_dir_var = tk.StringVar()
        self._status_var = tk.StringVar(value="Select a PDF to get started.")
        self._progress_var = tk.DoubleVar(value=0)

        self._build_widgets()
        self._populate_combos(self._languages)
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ #
    # Widget construction
    # ------------------------------------------------------------------ #

    def _build_widgets(self):
        padding = {"padx": 10, "pady": 5}

        # Input PDF
        input_frame = ttk.LabelFrame(self.root, text="Input PDF", padding=8)
        input_frame.grid(row=0, column=0, sticky="ew", **padding)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self._pdf_path_var, width=58).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(input_frame, text="Browse...", command=self._browse_pdf).grid(
            row=0, column=2
        )

        # Languages
        lang_frame = ttk.LabelFrame(self.root, text="Languages", padding=8)
        lang_frame.grid(row=1, column=0, sticky="ew", **padding)

        ttk.Label(lang_frame, text="From:").grid(row=0, column=0, sticky="w")
        self._src_combo = ttk.Combobox(
            lang_frame, textvariable=self._source_var, width=24, state="readonly"
        )
        self._src_combo.grid(row=0, column=1, padx=4)

        ttk.Label(lang_frame, text="  To:").grid(row=0, column=2, sticky="w")
        self._tgt_combo = ttk.Combobox(
            lang_frame, textvariable=self._target_var, width=24, state="readonly"
        )
        self._tgt_combo.grid(row=0, column=3, padx=4)

        # Output folder
        out_frame = ttk.LabelFrame(self.root, text="Output Folder", padding=8)
        out_frame.grid(row=2, column=0, sticky="ew", **padding)
        out_frame.columnconfigure(1, weight=1)

        ttk.Label(out_frame, text="Save to:").grid(row=0, column=0, sticky="w")
        ttk.Entry(out_frame, textvariable=self._output_dir_var, width=58).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).grid(
            row=0, column=2
        )

        # Translate button
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=3, column=0, pady=6)
        self._translate_btn = ttk.Button(
            btn_frame, text="Translate PDF", command=self._start_translation
        )
        self._translate_btn.pack()

        # Progress bar
        self._progress_bar = ttk.Progressbar(
            self.root,
            mode="determinate",
            maximum=100,
            variable=self._progress_var,
            length=680,
        )
        self._progress_bar.grid(row=4, column=0, padx=10, pady=4, sticky="ew")

        # Status label
        ttk.Label(
            self.root,
            textvariable=self._status_var,
            wraplength=680,
            anchor="w",
            foreground="gray",
        ).grid(row=5, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.root.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ #
    # Language loading
    # ------------------------------------------------------------------ #

    def _populate_combos(self, languages: list[dict]):
        self._languages = languages
        values = [f"{l['code']} — {l['name']}" for l in languages]
        self._src_combo["values"] = values
        self._tgt_combo["values"] = values

        # Set defaults
        self._set_combo_by_code(self._src_combo, self._source_var, "es")
        self._set_combo_by_code(self._tgt_combo, self._target_var, "en")

    def _set_combo_by_code(self, combo: ttk.Combobox, var: tk.StringVar, code: str):
        for i, lang in enumerate(self._languages):
            if lang["code"] == code:
                combo.current(i)
                var.set(code)
                return

    # ------------------------------------------------------------------ #
    # File pickers
    # ------------------------------------------------------------------ #

    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._pdf_path_var.set(path)
            if not self._output_dir_var.get():
                self._output_dir_var.set(os.path.dirname(path))

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._output_dir_var.set(folder)

    # ------------------------------------------------------------------ #
    # Translation job
    # ------------------------------------------------------------------ #

    def _start_translation(self):
        pdf_path = self._pdf_path_var.get().strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            messagebox.showerror("Error", "Please select a valid PDF file.")
            return

        # Resolve language codes from combo selection
        src_code = self._resolve_code(self._src_combo, self._source_var, "auto")
        tgt_code = self._resolve_code(self._tgt_combo, self._target_var, "en")

        if src_code == tgt_code and src_code != "auto":
            messagebox.showwarning("Warning", "Source and target languages are the same.")
            return

        output_dir = self._output_dir_var.get().strip() or os.path.dirname(pdf_path)

        self._translate_btn.config(state="disabled")
        self._progress_var.set(0)
        self._status_var.set("Starting translation...")

        threading.Thread(
            target=self._worker,
            args=(pdf_path, src_code, tgt_code, output_dir),
            daemon=True,
        ).start()

    def _resolve_code(self, combo: ttk.Combobox, var: tk.StringVar, default: str) -> str:
        try:
            idx = combo.current()
            if 0 <= idx < len(self._languages):
                return self._languages[idx]["code"]
        except Exception:
            pass
        return var.get() or default

    def _worker(self, pdf_path: str, source: str, target: str, output_dir: str):
        try:
            # Step 1: extract
            self.queue.put(("status", "Extracting text blocks from PDF..."))
            all_pages = extractor.extract_blocks(pdf_path)
            total_blocks = extractor.count_blocks(all_pages)

            if total_blocks == 0:
                self.queue.put((
                    "error",
                    "No text found in this PDF.\n\n"
                    "This may be a scanned document — OCR is required for those.",
                ))
                return

            # Step 2: load model (converts + caches on first run)
            self.queue.put(("status", "Loading optimized translation model..."))

            def _model_status(msg):
                self.queue.put(("status", msg))

            src_sp, tgt_sp, ct2_translator = tr.load_model(source, target, _model_status)

            # Step 3: translate in batches
            self.queue.put(("status", f"Translating {total_blocks} text blocks..."))
            flat_blocks = [b for page in all_pages for b in page]
            flat_texts = [b["text"] for b in flat_blocks]

            def _progress(current, total):
                pct = int(current / total * 100)
                self.queue.put(("progress", pct))
                self.queue.put(("status", f"Translating block {current} of {total}..."))

            flat_translated = tr.translate_batch(
                flat_texts, src_sp, tgt_sp, ct2_translator, _progress
            )

            # Re-group by page
            it = iter(flat_translated)
            translated_pages = [[next(it) for _ in page] for page in all_pages]

            # Step 4: build output PDF
            self.queue.put(("status", "Building translated PDF..."))
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(output_dir, f"{base}_translated.pdf")
            builder.build_translated_pdf(pdf_path, all_pages, translated_pages, output_path)

            self.queue.put(("done", output_path))

        except Exception as exc:
            self.queue.put(("error", str(exc)))

    # ------------------------------------------------------------------ #
    # Queue polling (main thread)
    # ------------------------------------------------------------------ #

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.queue.get_nowait()
                if msg_type == "status":
                    self._status_var.set(payload)
                elif msg_type == "progress":
                    self._progress_var.set(payload)
                elif msg_type == "done":
                    self._progress_var.set(100)
                    self._status_var.set(f"Done! Saved to: {payload}")
                    self._translate_btn.config(state="normal")
                    messagebox.showinfo("Done", f"Translation complete!\n\n{payload}")
                elif msg_type == "error":
                    self._status_var.set(f"Error: {payload}")
                    self._translate_btn.config(state="normal")
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)
