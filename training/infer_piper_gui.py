"""
Small desktop GUI for Piper inference with the trained local voice.

Examples:
  python training/infer_piper_gui.py
"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from dotenv import load_dotenv

from console_encoding import ensure_utf8_console
from infer_piper_text import _default_out_path, _play_wav, _resolve_model, synthesize_text

ensure_utf8_console()
load_dotenv()

BASE = Path(__file__).parent.parent
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_W = 0.8
DEFAULT_SENTENCE_SILENCE = 0.2


class PiperInferApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Piper Inference")
        self.root.geometry("920x700")

        self.model_var = tk.StringVar(value=str(_resolve_model("")))
        self.out_var = tk.StringVar(value=str(_default_out_path()))
        self.noise_scale_var = tk.StringVar(value=f"{DEFAULT_NOISE_SCALE:.3f}")
        self.length_scale_var = tk.StringVar(value=f"{DEFAULT_LENGTH_SCALE:.3f}")
        self.noise_w_var = tk.StringVar(value=f"{DEFAULT_NOISE_W:.3f}")
        self.noise_scale_slider_var = tk.DoubleVar(value=DEFAULT_NOISE_SCALE)
        self.length_scale_slider_var = tk.DoubleVar(value=DEFAULT_LENGTH_SCALE)
        self.noise_w_slider_var = tk.DoubleVar(value=DEFAULT_NOISE_W)
        self.sentence_silence_var = tk.StringVar(value=f"{DEFAULT_SENTENCE_SILENCE:.3f}")
        self.sentence_silence_slider_var = tk.DoubleVar(value=DEFAULT_SENTENCE_SILENCE)
        self.extra_args_var = tk.StringVar(value="")
        self.play_var = tk.BooleanVar(value=True)
        self.busy = False
        self.last_output: Path | None = None

        self._build()

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 6}
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(outer, text="Model").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(outer, textvariable=self.model_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(outer, text="Browse", command=self._browse_model).grid(row=0, column=2, **pad)

        ttk.Label(outer, text="Output WAV").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(outer, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(outer, text="Browse", command=self._browse_out).grid(row=1, column=2, **pad)

        flags = ttk.LabelFrame(outer, text="Flags", padding=8)
        flags.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        for idx in range(4):
            flags.columnconfigure(idx, weight=1 if idx == 1 else 0)

        ttk.Label(flags, text="noise_scale").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(
            flags,
            from_=0.1,
            to=1.5,
            variable=self.noise_scale_slider_var,
            command=lambda _v: self._sync_slider_to_entry(self.noise_scale_slider_var, self.noise_scale_var),
        ).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        noise_entry = ttk.Entry(flags, textvariable=self.noise_scale_var, width=10)
        noise_entry.grid(row=0, column=2, sticky="ew", padx=6, pady=4)
        noise_entry.bind("<FocusOut>", lambda _e: self._sync_entry_to_slider(self.noise_scale_var, self.noise_scale_slider_var))

        ttk.Label(flags, text="length_scale").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(
            flags,
            from_=0.5,
            to=1.8,
            variable=self.length_scale_slider_var,
            command=lambda _v: self._sync_slider_to_entry(self.length_scale_slider_var, self.length_scale_var),
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        length_entry = ttk.Entry(flags, textvariable=self.length_scale_var, width=10)
        length_entry.grid(row=1, column=2, sticky="ew", padx=6, pady=4)
        length_entry.bind("<FocusOut>", lambda _e: self._sync_entry_to_slider(self.length_scale_var, self.length_scale_slider_var))

        ttk.Label(flags, text="noise_w").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(
            flags,
            from_=0.1,
            to=1.5,
            variable=self.noise_w_slider_var,
            command=lambda _v: self._sync_slider_to_entry(self.noise_w_slider_var, self.noise_w_var),
        ).grid(row=2, column=1, sticky="ew", padx=6, pady=4)
        noise_w_entry = ttk.Entry(flags, textvariable=self.noise_w_var, width=10)
        noise_w_entry.grid(row=2, column=2, sticky="ew", padx=6, pady=4)
        noise_w_entry.bind("<FocusOut>", lambda _e: self._sync_entry_to_slider(self.noise_w_var, self.noise_w_slider_var))

        ttk.Label(flags, text="sentence_silence").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Scale(
            flags,
            from_=0.0,
            to=1.0,
            variable=self.sentence_silence_slider_var,
            command=lambda _v: self._sync_slider_to_entry(self.sentence_silence_slider_var, self.sentence_silence_var),
        ).grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        sentence_entry = ttk.Entry(flags, textvariable=self.sentence_silence_var, width=10)
        sentence_entry.grid(row=3, column=2, sticky="ew", padx=6, pady=4)
        sentence_entry.bind("<FocusOut>", lambda _e: self._sync_entry_to_slider(self.sentence_silence_var, self.sentence_silence_slider_var))

        ttk.Checkbutton(flags, text="Play after synth", variable=self.play_var).grid(row=0, column=3, sticky="e", padx=6, pady=4)
        ttk.Button(flags, text="Reset Defaults", command=self._reset_defaults).grid(row=1, column=3, sticky="e", padx=6, pady=4)

        ttk.Label(flags, text="Extra args").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(flags, textvariable=self.extra_args_var).grid(row=4, column=1, columnspan=3, sticky="ew", padx=6, pady=4)

        ttk.Label(outer, text="Prompt").grid(row=3, column=0, sticky="nw", **pad)
        self.text_box = tk.Text(outer, wrap=tk.WORD, height=14, font=("Segoe UI", 11))
        self.text_box.grid(row=3, column=1, columnspan=2, sticky="nsew", **pad)
        self.text_box.bind("<Control-v>", self._paste_prompt)
        self.text_box.bind("<Control-V>", self._paste_prompt)
        self.text_box.bind("<Shift-Insert>", self._paste_prompt)
        self.text_box.bind("<Button-3>", self._show_prompt_menu)
        self.prompt_menu = tk.Menu(self.root, tearoff=0)
        self.prompt_menu.add_command(label="Paste", command=self._paste_prompt)

        buttons = ttk.Frame(outer)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Button(buttons, text="Synthesize", command=self._start_synth).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Play Last WAV", command=self._play_last).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="New Output Path", command=self._refresh_out).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(outer, textvariable=self.status_var).grid(row=5, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Label(outer, text="Log").grid(row=6, column=0, sticky="nw", **pad)
        self.log_box = tk.Text(outer, wrap=tk.WORD, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.log_box.grid(row=6, column=1, columnspan=2, sticky="nsew", **pad)
        outer.rowconfigure(6, weight=1)

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Piper ONNX",
            initialdir=str(BASE / "output" / "piper_voice"),
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _browse_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save WAV As",
            defaultextension=".wav",
            initialdir=str(BASE / "output" / "piper_infer"),
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.out_var.set(path)

    def _refresh_out(self) -> None:
        self.out_var.set(str(_default_out_path()))

    def _sync_slider_to_entry(self, slider_var: tk.DoubleVar, entry_var: tk.StringVar) -> None:
        entry_var.set(f"{slider_var.get():.3f}")

    def _sync_entry_to_slider(self, entry_var: tk.StringVar, slider_var: tk.DoubleVar) -> None:
        raw = entry_var.get().strip()
        if not raw:
            return
        try:
            slider_var.set(float(raw))
            entry_var.set(f"{slider_var.get():.3f}")
        except ValueError:
            pass

    def _reset_defaults(self) -> None:
        self.noise_scale_slider_var.set(DEFAULT_NOISE_SCALE)
        self.length_scale_slider_var.set(DEFAULT_LENGTH_SCALE)
        self.noise_w_slider_var.set(DEFAULT_NOISE_W)
        self.sentence_silence_slider_var.set(DEFAULT_SENTENCE_SILENCE)
        self._sync_slider_to_entry(self.noise_scale_slider_var, self.noise_scale_var)
        self._sync_slider_to_entry(self.length_scale_slider_var, self.length_scale_var)
        self._sync_slider_to_entry(self.noise_w_slider_var, self.noise_w_var)
        self._sync_slider_to_entry(self.sentence_silence_slider_var, self.sentence_silence_var)

    def _paste_prompt(self, _event=None):
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            return "break"
        self.text_box.insert(tk.INSERT, text)
        return "break"

    def _show_prompt_menu(self, event):
        self.prompt_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _parse_optional_float(self, raw: str, label: str) -> float | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number") from exc

    def _start_synth(self) -> None:
        if self.busy:
            return

        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Piper Inference", "Type a prompt first.")
            return

        try:
            noise_scale = self._parse_optional_float(self.noise_scale_var.get(), "noise_scale")
            length_scale = self._parse_optional_float(self.length_scale_var.get(), "length_scale")
            noise_w = self._parse_optional_float(self.noise_w_var.get(), "noise_w")
            sentence_silence = self._parse_optional_float(self.sentence_silence_var.get(), "sentence_silence")
        except ValueError as exc:
            messagebox.showerror("Piper Inference", str(exc))
            return

        model = Path(self.model_var.get().strip())
        out_wav = Path(self.out_var.get().strip())
        if not model.exists():
            messagebox.showerror("Piper Inference", f"Model not found:\n{model}")
            return
        if not out_wav.is_absolute():
            out_wav = (BASE / out_wav).resolve()
            self.out_var.set(str(out_wav))
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        self.busy = True
        self.status_var.set("Synthesizing...")
        self._append_log(f"[RUN] model={model}")
        self._append_log(f"[RUN] out={out_wav}")

        def worker() -> None:
            try:
                result = synthesize_text(
                    text,
                    model,
                    out_wav,
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                    noise_w=noise_w,
                    sentence_silence=sentence_silence,
                    extra_args=self.extra_args_var.get(),
                )
                self.root.after(0, lambda: self._finish_synth(result, out_wav))
            except Exception as exc:
                self.root.after(0, lambda: self._fail_synth(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_synth(self, result, out_wav: Path) -> None:
        self.busy = False
        if result.returncode != 0:
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            self.status_var.set("Inference failed.")
            if stdout:
                self._append_log(stdout.strip())
            if stderr:
                self._append_log(stderr.strip())
            messagebox.showerror("Piper Inference", "Synthesis failed. See log.")
            return

        self.last_output = out_wav
        self.status_var.set(f"Done: {out_wav.name}")
        self._append_log(f"[OK] {out_wav}")
        if self.play_var.get():
            try:
                _play_wav(out_wav)
            except Exception as exc:
                self._append_log(f"[WARN] Playback failed: {exc}")
                messagebox.showwarning("Piper Inference", f"Synthesis worked, but playback failed:\n{exc}")

    def _fail_synth(self, message: str) -> None:
        self.busy = False
        self.status_var.set("Inference failed.")
        self._append_log(f"[ERROR] {message}")
        messagebox.showerror("Piper Inference", message)

    def _play_last(self) -> None:
        if not self.last_output or not self.last_output.exists():
            messagebox.showinfo("Piper Inference", "No synthesized WAV available yet.")
            return
        try:
            self.status_var.set(f"Playing: {self.last_output.name}")
            _play_wav(self.last_output)
            self.status_var.set("Ready.")
        except Exception as exc:
            messagebox.showerror("Piper Inference", f"Playback failed:\n{exc}")


def main() -> None:
    root = tk.Tk()
    root.option_add("*Font", "{Segoe UI} 10")
    PiperInferApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
