"""
Windows-friendly recorder for a human Greek voice dataset.

Features:
  - Reads prompts from `data/voice_recording_manifest.csv`
  - Works with USB microphones via `sounddevice`
  - Live input level meter (idle) uses the same gain as recording for threshold checks
  - Saves one WAV per sentence
  - Maintains Piper/LJSpeech-style `metadata.csv`
  - Optional prompt playback via Windows SAPI
  - Optional "Next: skip to next missing take"; otherwise Next is always one sentence forward

Run:
  python training/record_human_voice_dataset.py

List devices only:
  python training/record_human_voice_dataset.py --list-devices
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import messagebox, ttk


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
MANIFEST_PATH = DATA_DIR / "voice_recording_manifest.csv"
OUT_DIR = DATA_DIR / "human_voice_dataset"
WAV_DIR = OUT_DIR / "wavs"
METADATA_PATH = OUT_DIR / "metadata.csv"
STATE_PATH = OUT_DIR / "recording_state.json"

SAMPLE_RATE = 22050
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 2048
# Larger PortAudio buffer reduces input_overflow when the Python/Tk main thread is busy.
STREAM_LATENCY = "high"
# When "Read prompt aloud" runs at Arm time, skip capturing this long so speaker bleed is not recorded.
PROMPT_TTS_BLEED_SKIP_SEC = 0.35

# Sensible defaults for a typical quiet room + USB headset mic (tweak per setup).
DEFAULT_INPUT_GAIN = 1.0
INPUT_GAIN_MIN = 0.05
INPUT_GAIN_MAX = 4.0
DEFAULT_SPEECH_THRESHOLD = 0.001
DEFAULT_TRAILING_SILENCE_SEC = 0.85
DEFAULT_MIN_DURATION_SEC = 1.0
DEFAULT_MAX_DURATION_SEC = 12.0
# After trimming, fade in the first samples to avoid a click when PCM starts non-zero.
LEAD_IN_FADE_SEC = 0.005
# Progress bar caps here so loud peaks still show a full bar without raising widget maximum.
LEVEL_BAR_CAP = 0.05
# Subtract mean of the opening window to tame DC offset "thump" at playback start.
DC_WINDOW_SEC = 0.03


@dataclass
class PromptItem:
    utterance_id: str
    category: str
    text: str


def list_input_devices() -> list[dict[str, Any]]:
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": index,
                    "name": device["name"],
                    "hostapi": sd.query_hostapis(device["hostapi"])["name"],
                    "max_input_channels": device["max_input_channels"],
                    "default_samplerate": device["default_samplerate"],
                }
            )
    return devices


def print_input_devices() -> None:
    default_input = sd.default.device[0]
    print(f"Default input device: {default_input}")
    for device in list_input_devices():
        print(
            f"{device['index']}: {device['name']} | hostapi={device['hostapi']} | "
            f"in={device['max_input_channels']} | sr={device['default_samplerate']}"
        )


def load_manifest(path: Path) -> list[PromptItem]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt manifest not found: {path}")

    items: list[PromptItem] = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            utterance_id = (row.get("utterance_id") or "").strip()
            category = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if not utterance_id or not text:
                continue
            items.append(PromptItem(utterance_id=utterance_id, category=category, text=text))
    if not items:
        raise ValueError(f"No prompt rows found in {path}")
    return items


class RecorderApp:
    def __init__(self, root: tk.Tk, prompts: list[PromptItem]) -> None:
        self.root = root
        self.prompts = prompts
        self.root.title("Gemma4GR Human Voice Recorder")
        self.root.geometry("1040x720")

        WAV_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.devices = list_input_devices()
        if not self.devices:
            raise RuntimeError("No input devices found.")

        self.device_map = {
            f"{d['index']}: {d['name']} [{d['hostapi']}]": d["index"] for d in self.devices
        }
        self.current_index = self._load_saved_index()

        self.stream: sd.InputStream | None = None
        self.meter_stream: sd.InputStream | None = None
        self.stream_lock = threading.Lock()
        self.status_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.recording_active = False
        self.speech_started = False
        self.pending_auto_stop = False
        self.pending_manual_stop = False
        self._discard_audio_until_monotonic: float | None = None

        self.preroll_chunks: deque[np.ndarray] = deque()
        self.recorded_chunks: list[np.ndarray] = []
        self.latest_level = 0.0
        self.silence_samples = 0
        self.recorded_samples = 0
        self.max_preroll_chunks = 8

        self.device_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="")
        self.level_var = tk.DoubleVar(value=0.0)
        self.level_text_var = tk.StringVar(value="RMS: 0.000000")
        self.auto_advance_var = tk.BooleanVar(value=True)
        self.auto_arm_next_var = tk.BooleanVar(value=True)
        self.read_prompt_var = tk.BooleanVar(value=False)
        self.normalize_audio_var = tk.BooleanVar(value=True)
        self.skip_existing_var = tk.BooleanVar(value=False)

        self.input_gain_var = tk.DoubleVar(value=DEFAULT_INPUT_GAIN)
        self.threshold_var = tk.DoubleVar(value=DEFAULT_SPEECH_THRESHOLD)
        self.trailing_silence_var = tk.DoubleVar(value=DEFAULT_TRAILING_SILENCE_SEC)
        self.min_duration_var = tk.DoubleVar(value=DEFAULT_MIN_DURATION_SEC)
        self.max_duration_var = tk.DoubleVar(value=DEFAULT_MAX_DURATION_SEC)
        self.gain_value_label: ttk.Label | None = None
        self.threshold_value_label: ttk.Label | None = None
        self.trailing_value_label: ttk.Label | None = None
        self.min_duration_value_label: ttk.Label | None = None
        self.max_duration_value_label: ttk.Label | None = None

        self._last_overflow_notice_monotonic = 0.0
        self._finalize_running = False
        self._arm_after_id: str | None = None

        self._build_ui()
        self._select_default_device()
        self._refresh_prompt_view()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(80, self._poll_status_queue)
        self.root.after(120, self._refresh_numeric_labels)
        self.root.after(150, self._start_meter_stream)

    def _maybe_queue_overflow_notice(self, context: str) -> None:
        now = time.monotonic()
        if now - self._last_overflow_notice_monotonic < 3.0:
            return
        self._last_overflow_notice_monotonic = now
        self.status_queue.put(
            (
                "status",
                f"Input overflow ({context}). Larger buffers are enabled; if this repeats, "
                "close other apps using the microphone or switch Windows recording device format/WASAPI.",
            )
        )

    def _gain_mono_and_rms(self, indata: np.ndarray) -> tuple[np.ndarray, float]:
        mono = np.mean(indata[:, :CHANNELS], axis=1).astype(np.float32).copy()
        gain = float(self.input_gain_var.get())
        mono = np.clip(mono * gain, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-12)
        return mono, rms

    def _mono_gain_rms(self, indata: np.ndarray) -> float:
        _, rms = self._gain_mono_and_rms(indata)
        return rms

    def _stop_meter_stream(self) -> None:
        to_close: sd.InputStream | None = None
        with self.stream_lock:
            to_close = self.meter_stream
            self.meter_stream = None
        if to_close is not None:
            try:
                to_close.stop()
            except Exception:
                pass
            try:
                to_close.close()
            except Exception:
                pass

    def _start_meter_stream(self) -> None:
        if self.recording_active:
            return
        self._stop_meter_stream()
        try:
            device_index = self.selected_device_index()
        except RuntimeError:
            return

        def meter_callback(indata: np.ndarray, _frames: int, _time_info: Any, status: sd.CallbackFlags) -> None:
            if status:
                if getattr(status, "input_overflow", False):
                    self._maybe_queue_overflow_notice("idle meter")
                else:
                    self.status_queue.put(("status", f"Audio callback status: {status}"))
            self.latest_level = self._mono_gain_rms(indata)

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                device=device_index,
                channels=CHANNELS,
                dtype=DTYPE,
                latency=STREAM_LATENCY,
                callback=meter_callback,
            )
            with self.stream_lock:
                self.meter_stream = stream
            stream.start()
        except Exception:
            self._stop_meter_stream()
            self.status_var.set("Idle level meter could not open this input device.")

    def _on_input_device_selected(self, _evt: Any = None) -> None:
        if self.recording_active:
            return
        self._start_meter_stream()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        ttk.Label(top, text="Input Device").grid(row=0, column=0, sticky="w")
        self.device_combo = ttk.Combobox(
            top,
            textvariable=self.device_var,
            values=list(self.device_map.keys()),
            state="readonly",
            width=70,
        )
        self.device_combo.grid(row=0, column=1, sticky="ew", padx=(8, 10))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_input_device_selected)
        ttk.Button(top, text="Refresh Devices", command=self.refresh_devices).grid(row=0, column=2)
        top.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(outer, text="Recording Controls", padding=10)
        settings.pack(fill="x", pady=(12, 0))

        ttk.Label(settings, text="Mic sensitivity (input gain)").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            settings,
            from_=INPUT_GAIN_MIN,
            to=INPUT_GAIN_MAX,
            variable=self.input_gain_var,
            orient="horizontal",
        ).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        self.gain_value_label = ttk.Label(settings, text="")
        self.gain_value_label.grid(row=0, column=2, sticky="w")

        ttk.Label(settings, text="Speech threshold").grid(row=1, column=0, sticky="w")
        ttk.Scale(settings, from_=0.00005, to=0.02, variable=self.threshold_var, orient="horizontal").grid(
            row=1, column=1, sticky="ew", padx=8
        )
        self.threshold_value_label = ttk.Label(settings, text="")
        self.threshold_value_label.grid(row=1, column=2, sticky="w")

        ttk.Label(settings, text="Trailing silence (sec)").grid(row=2, column=0, sticky="w")
        ttk.Scale(
            settings, from_=0.30, to=2.50, variable=self.trailing_silence_var, orient="horizontal"
        ).grid(row=2, column=1, sticky="ew", padx=8)
        self.trailing_value_label = ttk.Label(settings, text="")
        self.trailing_value_label.grid(row=2, column=2, sticky="w")

        ttk.Label(settings, text="Min duration (sec)").grid(row=3, column=0, sticky="w")
        ttk.Scale(settings, from_=0.5, to=3.0, variable=self.min_duration_var, orient="horizontal").grid(
            row=3, column=1, sticky="ew", padx=8
        )
        self.min_duration_value_label = ttk.Label(settings, text="")
        self.min_duration_value_label.grid(row=3, column=2, sticky="w")

        ttk.Label(settings, text="Max duration (sec)").grid(row=4, column=0, sticky="w")
        ttk.Scale(settings, from_=4.0, to=20.0, variable=self.max_duration_var, orient="horizontal").grid(
            row=4, column=1, sticky="ew", padx=8
        )
        self.max_duration_value_label = ttk.Label(settings, text="")
        self.max_duration_value_label.grid(row=4, column=2, sticky="w")

        tip = (
            "Suggested starting point: gain 1.0; threshold ~0.001 while watching RMS; trailing silence "
            "~0.85 s; min duration ~1 s; max ~12 s. Raise gain if RMS is low; lower gain if the meter "
            "maxes out or sound clips; lower threshold if arming waits too long."
        )
        ttk.Label(settings, text=tip, wraplength=880, justify="left").grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )

        ttk.Checkbutton(settings, text="Auto advance after save", variable=self.auto_advance_var).grid(
            row=0, column=3, padx=(14, 0), sticky="w"
        )
        ttk.Checkbutton(settings, text="Auto arm next sentence", variable=self.auto_arm_next_var).grid(
            row=1, column=3, padx=(14, 0), sticky="w"
        )
        ttk.Checkbutton(settings, text="Read prompt aloud", variable=self.read_prompt_var).grid(
            row=2, column=3, padx=(14, 0), sticky="w"
        )
        ttk.Checkbutton(settings, text="Normalize saved audio", variable=self.normalize_audio_var).grid(
            row=3, column=3, padx=(14, 0), sticky="w"
        )
        ttk.Checkbutton(
            settings,
            text="Next: skip to next missing take (else always one sentence forward)",
            variable=self.skip_existing_var,
        ).grid(row=4, column=3, padx=(14, 0), sticky="w")
        settings.columnconfigure(1, weight=1)

        meter = ttk.LabelFrame(outer, text="Input Level (live preview, same gain as recording)", padding=10)
        meter.pack(fill="x", pady=(12, 0))
        self.level_bar = ttk.Progressbar(
            meter, mode="determinate", maximum=LEVEL_BAR_CAP, variable=self.level_var
        )
        self.level_bar.pack(fill="x")
        ttk.Label(meter, textvariable=self.level_text_var).pack(anchor="w", pady=(6, 0))

        prompt_box = ttk.LabelFrame(outer, text="Prompt", padding=10)
        prompt_box.pack(fill="both", expand=True, pady=(12, 0))

        self.progress_label = ttk.Label(prompt_box, textvariable=self.progress_var, font=("Segoe UI", 10, "bold"))
        self.progress_label.pack(anchor="w")

        self.category_label = ttk.Label(prompt_box, text="", font=("Segoe UI", 10))
        self.category_label.pack(anchor="w", pady=(6, 8))

        self.prompt_text = tk.Text(prompt_box, wrap="word", height=10, font=("Segoe UI", 18))
        self.prompt_text.pack(fill="both", expand=True)
        self.prompt_text.configure(state="disabled")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(12, 0))

        ttk.Button(buttons, text="Previous", command=self.previous_prompt).pack(side="left")
        ttk.Button(buttons, text="Next", command=self.next_prompt).pack(side="left", padx=(8, 0))
        ttk.Label(buttons, text="Go to #").pack(side="left", padx=(16, 4))
        self.jump_entry = ttk.Entry(buttons, width=6)
        self.jump_entry.pack(side="left")
        ttk.Button(buttons, text="Go", command=self.jump_to_sentence).pack(side="left", padx=(4, 0))
        self.jump_entry.bind("<Return>", lambda _e: self.jump_to_sentence())
        ttk.Button(buttons, text="Speak Prompt", command=self.speak_prompt).pack(side="left", padx=(16, 0))
        ttk.Button(buttons, text="Open Output Folder", command=self.open_output_folder).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Arm / Record", command=self.start_recording).pack(side="right")
        ttk.Button(buttons, text="Stop", command=self.stop_recording).pack(side="right", padx=(0, 8))
        ttk.Button(buttons, text="Retry Sentence", command=self.retry_current).pack(side="right", padx=(0, 8))

        status = ttk.LabelFrame(outer, text="Status", padding=10)
        status.pack(fill="x", pady=(12, 0))
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

    def _select_default_device(self) -> None:
        self.device_combo.current(1)

    def _load_saved_index(self) -> int:
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                idx = int(data.get("current_index", 0))
                if 0 <= idx < len(self.prompts):
                    return idx
            except Exception:
                pass

        completed = 0
        for prompt in self.prompts:
            if (WAV_DIR / f"{prompt.utterance_id}.wav").exists():
                completed += 1
            else:
                break
        return min(completed, len(self.prompts) - 1)

    def _save_state(self) -> None:
        payload = {"current_index": self.current_index, "saved_at": time.time()}
        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _refresh_prompt_view(self) -> None:
        prompt = self.prompts[self.current_index]
        recorded = sum(1 for item in self.prompts if (WAV_DIR / f"{item.utterance_id}.wav").exists())
        self.progress_var.set(
            f"Sentence {self.current_index + 1}/{len(self.prompts)} | Recorded {recorded}/{len(self.prompts)} | ID {prompt.utterance_id}"
        )
        self.category_label.config(text=f"Category: {prompt.category}")
        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", prompt.text)
        self.prompt_text.configure(state="disabled")
        self._save_state()

    def current_prompt(self) -> PromptItem:
        return self.prompts[self.current_index]

    def current_wav_path(self) -> Path:
        return WAV_DIR / f"{self.current_prompt().utterance_id}.wav"

    def refresh_devices(self) -> None:
        self.devices = list_input_devices()
        self.device_map = {
            f"{d['index']}: {d['name']} [{d['hostapi']}]": d["index"] for d in self.devices
        }
        self.device_combo["values"] = list(self.device_map.keys())
        self._select_default_device()
        self.status_var.set("Input device list refreshed.")
        self._start_meter_stream()

    def selected_device_index(self) -> int:
        label = self.device_var.get()
        if label not in self.device_map:
            raise RuntimeError("Select a valid input device first.")
        return self.device_map[label]

    def _cancel_pending_arm(self) -> None:
        if self._arm_after_id is not None:
            try:
                self.root.after_cancel(self._arm_after_id)
            except Exception:
                pass
            self._arm_after_id = None

    def _schedule_auto_arm(self) -> None:
        self._cancel_pending_arm()

        def _arm() -> None:
            self._arm_after_id = None
            if not self.recording_active:
                self.start_recording()

        self._arm_after_id = self.root.after(500, _arm)

    def jump_to_sentence(self) -> None:
        if self.recording_active:
            return
        raw = self.jump_entry.get().strip()
        if not raw:
            self.status_var.set("Enter a sentence number (1-based), then Go or Enter.")
            return
        try:
            n = int(raw)
        except ValueError:
            self.status_var.set(f"Not a number: {raw!r}")
            return
        if n < 1 or n > len(self.prompts):
            self.status_var.set(f"Sentence # must be between 1 and {len(self.prompts)}.")
            return
        self._cancel_pending_arm()
        self.current_index = n - 1
        self._refresh_prompt_view()
        self.status_var.set(f"Jumped to sentence {n}.")

    def previous_prompt(self) -> None:
        if self.recording_active:
            return
        self._cancel_pending_arm()
        self.current_index = max(0, self.current_index - 1)
        self._refresh_prompt_view()

    def next_prompt(self) -> None:
        if self.recording_active:
            return
        self._cancel_pending_arm()
        if self.skip_existing_var.get():
            self._move_next_missing_take()
        else:
            if self.current_index >= len(self.prompts) - 1:
                self.status_var.set("Already at the last prompt.")
                return
            self.current_index += 1
            self._refresh_prompt_view()

    def _move_next_missing_take(self) -> None:
        """First index after current that has no WAV; if none, step by one for redos."""
        if self.current_index >= len(self.prompts) - 1:
            self.status_var.set("Already at the last prompt.")
            return
        next_index = self.current_index + 1
        while next_index < len(self.prompts):
            if not (WAV_DIR / f"{self.prompts[next_index].utterance_id}.wav").exists():
                self.current_index = next_index
                self._refresh_prompt_view()
                self.status_var.set("Next missing take.")
                return
            next_index += 1
        self.current_index += 1
        self._refresh_prompt_view()
        self.status_var.set("Next sentence (take exists on disk; re-record to replace).")

    def speak_prompt(self) -> None:
        prompt = self.current_prompt().text

        def worker() -> None:
            escaped = prompt.replace("'", "''")
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{escaped}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        threading.Thread(target=worker, daemon=True).start()

    def open_output_folder(self) -> None:
        subprocess.run(["explorer", str(OUT_DIR)], check=False)

    def retry_current(self) -> None:
        if self.recording_active:
            return
        self._cancel_pending_arm()
        path = self.current_wav_path()
        if path.exists():
            path.unlink()
        self.rewrite_metadata()
        self.status_var.set(f"Deleted current take for {self.current_prompt().utterance_id}.")

    def start_recording(self) -> None:
        if self.recording_active:
            return

        try:
            device_index = self.selected_device_index()
        except Exception as exc:
            messagebox.showerror("Recorder", str(exc))
            return

        self._stop_meter_stream()

        self.preroll_chunks = deque(maxlen=self.max_preroll_chunks)
        self.recorded_chunks = []
        self.latest_level = 0.0
        self.silence_samples = 0
        self.recorded_samples = 0
        self.speech_started = False
        self.pending_auto_stop = False
        self.pending_manual_stop = False
        self._discard_audio_until_monotonic = (
            time.monotonic() + PROMPT_TTS_BLEED_SKIP_SEC if self.read_prompt_var.get() else None
        )

        def callback(indata: np.ndarray, _frames: int, _time_info: Any, status: sd.CallbackFlags) -> None:
            if status:
                if getattr(status, "input_overflow", False):
                    self._maybe_queue_overflow_notice("recording")
                else:
                    self.status_queue.put(("status", f"Audio callback status: {status}"))

            mono, rms = self._gain_mono_and_rms(indata)
            self.latest_level = rms

            bleed_cutoff = self._discard_audio_until_monotonic
            if bleed_cutoff is not None:
                if time.monotonic() < bleed_cutoff:
                    if not self.speech_started:
                        self.preroll_chunks.clear()
                    return
                self._discard_audio_until_monotonic = None

            threshold = float(self.threshold_var.get())
            if not self.speech_started:
                if rms >= threshold:
                    self.speech_started = True
                    self.recorded_chunks.extend(list(self.preroll_chunks))
                    self.recorded_chunks.append(mono)
                    self.recorded_samples = sum(len(chunk) for chunk in self.recorded_chunks)
                    self.silence_samples = 0
                    self.status_queue.put(("status", "Speech detected. Recording..."))
                    return
                self.preroll_chunks.append(mono)
                return

            self.recorded_chunks.append(mono)
            self.recorded_samples += len(mono)

            if rms >= threshold:
                self.silence_samples = 0
            else:
                self.silence_samples += len(mono)

            min_samples = int(float(self.min_duration_var.get()) * SAMPLE_RATE)
            max_samples = int(float(self.max_duration_var.get()) * SAMPLE_RATE)
            trailing_silence_samples = int(float(self.trailing_silence_var.get()) * SAMPLE_RATE)

            if self.recorded_samples >= max_samples:
                self.pending_auto_stop = True
                self.status_queue.put(("status", "Max duration reached. Saving take..."))
            elif self.recorded_samples >= min_samples and self.silence_samples >= trailing_silence_samples:
                self.pending_auto_stop = True
                self.status_queue.put(("status", "Trailing silence detected. Saving take..."))

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                device=device_index,
                channels=CHANNELS,
                dtype=DTYPE,
                latency=STREAM_LATENCY,
                callback=callback,
            )
            self.stream.start()
            self.recording_active = True
            self.status_var.set("Armed. Waiting for speech...")
            if self.read_prompt_var.get():
                self.speak_prompt()
        except Exception as exc:
            self.stream = None
            self.recording_active = False
            self._start_meter_stream()
            messagebox.showerror("Recorder", f"Could not start recording:\n{exc}")

    def stop_recording(self) -> None:
        if not self.recording_active:
            return
        self.pending_manual_stop = True
        self.status_var.set("Stopping recording...")

    def _finalize_recording(self) -> None:
        if self._finalize_running:
            return
        if not self.recording_active:
            return
        self._finalize_running = True
        try:
            with self.stream_lock:
                if self.stream is not None:
                    try:
                        self.stream.stop()
                    except Exception:
                        pass
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

            self.recording_active = False
            self.pending_auto_stop = False
            self.pending_manual_stop = False

            try:
                if not self.speech_started or not self.recorded_chunks:
                    self.status_var.set("No speech detected. Nothing saved.")
                    return

                audio = np.concatenate(self.recorded_chunks).astype(np.float32)
                audio = self.trim_silence(audio, float(self.threshold_var.get()) * 0.60)
                if len(audio) < int(0.35 * SAMPLE_RATE):
                    self.status_var.set("Take too short after trimming. Nothing saved.")
                    return

                audio = self._de_click_saved_clip(audio)

                if self.normalize_audio_var.get():
                    peak = float(np.max(np.abs(audio)) + 1e-12)
                    if peak > 0:
                        audio = np.clip(audio * (0.95 / peak), -1.0, 1.0)

                wav_path = self.current_wav_path()
                sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
                self.rewrite_metadata()
                self.status_var.set(f"Saved {wav_path.name}")

                if self.auto_advance_var.get():
                    if self.current_index < len(self.prompts) - 1:
                        self.current_index += 1
                        self._refresh_prompt_view()
                    else:
                        self.status_var.set("Saved. Reached the last prompt.")
                    if self.auto_arm_next_var.get():
                        self._schedule_auto_arm()
            finally:
                self.recorded_chunks = []
                self.preroll_chunks.clear()
                self.speech_started = False
                self.recorded_samples = 0
                self.silence_samples = 0
                self._start_meter_stream()
        finally:
            self._finalize_running = False

    def trim_silence(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        window = 256
        abs_audio = np.abs(audio)
        start = 0
        end = len(audio)

        while start + window < len(audio) and np.max(abs_audio[start : start + window]) < threshold:
            start += window
        while end - window > start and np.max(abs_audio[end - window : end]) < threshold:
            end -= window

        pad = int(0.08 * SAMPLE_RATE)
        start = max(0, start - pad)
        end = min(len(audio), end + pad)
        return audio[start:end]

    def _de_click_saved_clip(self, audio: np.ndarray) -> np.ndarray:
        """Trim leaves a hard edge at sample 0 and optional DC offset; tame playback clicks."""
        out = audio.astype(np.float32, copy=True)
        w = min(len(out), int(DC_WINDOW_SEC * SAMPLE_RATE))
        if w > 0:
            out -= np.float32(np.mean(out[:w]))
        fade_n = min(len(out), max(2, int(LEAD_IN_FADE_SEC * SAMPLE_RATE)))
        if fade_n > 1:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            out[:fade_n] *= ramp
        return out

    def rewrite_metadata(self) -> None:
        rows = []
        for item in self.prompts:
            wav_path = WAV_DIR / f"{item.utterance_id}.wav"
            if wav_path.exists():
                rows.append((item.utterance_id, item.text))

        with open(METADATA_PATH, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="|")
            for utterance_id, text in rows:
                writer.writerow([utterance_id, text])

    def _poll_status_queue(self) -> None:
        while True:
            try:
                event, payload = self.status_queue.get_nowait()
            except queue.Empty:
                break

            if event == "status":
                self.status_var.set(str(payload))

        cap = LEVEL_BAR_CAP
        self.level_var.set(min(float(self.latest_level), cap))

        if self.recording_active and (self.pending_auto_stop or self.pending_manual_stop):
            self.pending_auto_stop = False
            self.pending_manual_stop = False
            self._finalize_recording()

        self.root.after(80, self._poll_status_queue)

    def _refresh_numeric_labels(self) -> None:
        self.level_text_var.set(f"RMS: {self.latest_level:.6f}")
        if self.gain_value_label is not None:
            gv = float(self.input_gain_var.get())
            self.gain_value_label.config(text=f"{gv:.3f}x" if gv < 1.0 else f"{gv:.2f}x")
        if self.threshold_value_label is not None:
            self.threshold_value_label.config(text=f"{self.threshold_var.get():.5f}")
        if self.trailing_value_label is not None:
            self.trailing_value_label.config(text=f"{self.trailing_silence_var.get():.2f}")
        if self.min_duration_value_label is not None:
            self.min_duration_value_label.config(text=f"{self.min_duration_var.get():.2f}")
        if self.max_duration_value_label is not None:
            self.max_duration_value_label.config(text=f"{self.max_duration_var.get():.2f}")
        self.root.after(120, self._refresh_numeric_labels)

    def on_close(self) -> None:
        self._cancel_pending_arm()
        if self.recording_active:
            self.pending_manual_stop = True
            self._finalize_recording()
        self._stop_meter_stream()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="Print Windows input devices and exit")
    parser.add_argument("--manifest", type=Path, default=None, help="Override prompt manifest CSV (default: data/voice_recording_manifest.csv)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory for WAVs and metadata (default: data/human_voice_dataset)")
    args = parser.parse_args()

    if args.list_devices:
        print_input_devices()
        return

    manifest_path = args.manifest or MANIFEST_PATH
    if args.out_dir:
        global OUT_DIR, WAV_DIR, METADATA_PATH, STATE_PATH
        OUT_DIR = args.out_dir.resolve()
        WAV_DIR = OUT_DIR / "wavs"
        METADATA_PATH = OUT_DIR / "metadata.csv"
        STATE_PATH = OUT_DIR / "recording_state.json"

    prompts = load_manifest(manifest_path)
    root = tk.Tk()
    app = RecorderApp(root, prompts)
    root.mainloop()


if __name__ == "__main__":
    main()
