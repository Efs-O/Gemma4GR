"""
Gemma4GR — Greek Language Fine-tuning Pipeline
Entry point: python menu.py
"""
import sys, os, json, subprocess, webbrowser
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from env_bootstrap import select_python_for_script

from training.console_encoding import ensure_utf8_console

ensure_utf8_console()

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

BASE = Path(__file__).parent

COLAB_E2B    = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Audio.ipynb"
COLAB_E4B    = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Audio.ipynb"
COLAB_STUDIO = "https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb"

# Phase 1 — STT pipeline
PHASE1_STEPS = [
    ("1", "Setup Assets",               "training/setup_assets.py",         "assets/piper/piper.exe"),
    ("1p", "Setup Piper Sources (git)",  "training/setup_piper_sources.py",  "models/piper-src/.git"),
    ("2", "Download Moira Model",        "training/download_moira.py",       "models/moira/.downloaded"),
    ("3", "Generate Greek Audio Pairs",  "training/generate_pairs.py",       "data/raw_audio"),
    ("4", "Prepare STT Dataset",         "training/prepare_stt_dataset.py",  "data/train_stt.jsonl"),
    ("5", "Prepare Piper Dataset",       "training/prepare_piper_dataset.py","data/piper_dataset/metadata.csv"),
    ("6", "Benchmark BEFORE Training",   "tests/run_before.py",              "tests/benchmark_results/base_qa.json"),
    ("7", "Train E2B STT — Local GPU",   "training/train_e2b_local.py",      "output/e2b_greek_stt/lora_adapter"),
    ("9", "Train Piper Voice (Docker)",  "training/train_piper.py",          "output/piper_voice/el_GR-joy-medium.onnx"),
]

# Phase 2 — Greek Q&A text pipeline
PHASE2_STEPS = [
    ("D", "Generate Greek Q&A Pairs",   "training/generate_qa_pipeline.py",  "data/qa_pairs.jsonl"),
    ("DD", "Phase 2 Full Auto",         "training/run_phase2_auto.py",       "data/train_qa.jsonl"),
    ("E", "Prepare Q&A Dataset",         "training/prepare_qa_dataset.py",   "data/train_qa.jsonl"),
    ("F", "Train QA LoRA — Local GPU",   "training/train_qa_local.py",       "output/e2b_greek_qa/lora_adapter"),
    ("G", "Merge Adapters → GGUF",       "training/merge_adapters.py",       r"N:\.cache\huggingface\hub\gemma4gr-e2b"),
]

# Results & utilities (shared)
UTIL_STEPS = [
    ("A", "Benchmark AFTER Training",    "tests/run_after.py",               "tests/benchmark_results/finetuned_qa.json"),
    ("B", "Compare Before/After",        None,                               None),
    ("C", "View Logs",                   None,                               None),
    ("S", "Validate Samples",            "training/validate_samples.py",     None),
]

# Backwards-compat: combined list for status checks
STEPS = PHASE1_STEPS + PHASE2_STEPS + UTIL_STEPS

COLAB_ITEMS = [
    ("8a", "Open E2B STT Colab",     COLAB_E2B),
    ("8b", "Open E4B STT Colab",     COLAB_E4B),
    ("8c", "Open Unsloth Studio",    COLAB_STUDIO),
    ("8d", "Open E4B Q&A Colab",     None),   # local notebook
]


def step_status(artifact: str | None) -> Text:
    if artifact is None:
        return Text("—", style="dim")
    artifact_path = Path(BASE / artifact)
    if artifact_path.is_dir():
        has_content = any(artifact_path.iterdir())
        if has_content:
            return Text("✓ Done", style="bold green")
        return Text("Pending", style="dim")
    if artifact_path.exists():
        return Text("✓ Done", style="bold green")
    return Text("Pending", style="dim")


def show_menu():
    console.clear()
    console.print(Rule("[bold blue]Gemma4GR — Greek Language Fine-tuning[/bold blue]"))

    # Phase 1 table
    t1 = Table(box=box.ROUNDED, border_style="blue", show_header=True, header_style="bold cyan")
    t1.add_column("Key", width=5)
    t1.add_column("Phase 1 — Greek STT", min_width=35)
    t1.add_column("Status", width=10)
    for key, label, _, artifact in PHASE1_STEPS:
        t1.add_row(f"[cyan]{key}[/cyan]", label, step_status(artifact))
    console.print(t1)

    # Phase 2 table
    t2 = Table(box=box.ROUNDED, border_style="green", show_header=True, header_style="bold green")
    t2.add_column("Key", width=5)
    t2.add_column("Phase 2 — Greek Q&A Text", min_width=35)
    t2.add_column("Status", width=10)
    for key, label, _, artifact in PHASE2_STEPS:
        t2.add_row(f"[green]{key}[/green]", label, step_status(artifact))
    console.print(t2)

    # Util table
    tu = Table(box=box.SIMPLE, border_style="dim", show_header=False)
    tu.add_column("Key", width=5)
    tu.add_column("Utility", min_width=35)
    tu.add_column("Status", width=10)
    for key, label, _, artifact in UTIL_STEPS:
        tu.add_row(f"[dim]{key}[/dim]", label, step_status(artifact))
    console.print(tu)

    # Colab section
    c = Table(box=box.SIMPLE, border_style="yellow", show_header=False)
    c.add_column("Key", width=5)
    c.add_column("Action", min_width=35)
    for key, label, _ in COLAB_ITEMS:
        c.add_row(f"[yellow]{key}[/yellow]", label)

    console.print(Panel(c, title="[yellow]Google Colab[/yellow]", expand=False))
    console.print(Rule())
    console.print("  [dim]Q[/dim]  Quit\n")


def run_script(script: str, log_name: str):
    script_path = BASE / script
    if not script_path.exists():
        console.print(f"[red]Script not found:[/red] {script_path}")
        input("\nPress Enter to return...")
        return

    log_path = BASE / "logs" / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(exist_ok=True)

    console.print(f"\n[yellow]▶ Running:[/yellow] {script}")
    console.print(f"[dim]Log:[/dim] {log_path}\n")
    console.print(Rule(style="dim"))

    env = os.environ.copy()
    python_exe, reason = select_python_for_script(BASE, script_path)
    if reason:
        console.print(f"[dim]Python:[/dim] {reason}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            stripped = line.rstrip()
            console.print(stripped)
            log_f.write(line)
        proc.wait()

    console.print(Rule(style="dim"))
    if proc.returncode == 0:
        console.print("\n[bold green]✓ Completed successfully.[/bold green]")
    else:
        console.print(f"\n[bold red]✗ Failed (exit {proc.returncode}). Check log:[/bold red] {log_path}")

    input("\nPress Enter to return to menu...")


def show_compare():
    console.clear()
    console.print(Rule("[bold green]Before vs After — Results[/bold green]"))

    results_dir = BASE / "tests" / "benchmark_results"
    metrics = []

    for prefix, label in [("base", "Base Model"), ("finetuned", "Fine-tuned")]:
        qa_f   = results_dir / f"{prefix}_qa.json"
        stt_f  = results_dir / f"{prefix}_stt.json"
        entry  = {"label": label, "qa": None, "wer": None, "cer": None}
        if qa_f.exists():
            entry["qa"] = json.loads(qa_f.read_text(encoding="utf-8")).get("score_pct")
        if stt_f.exists():
            d = json.loads(stt_f.read_text(encoding="utf-8"))
            entry["wer"] = d.get("WER")
            entry["cer"] = d.get("CER")
        metrics.append(entry)

    if not any(m["qa"] is not None for m in metrics):
        console.print("[yellow]No benchmark results found yet. Run steps 6 and A first.[/yellow]")
        input("\nPress Enter...")
        return

    t = Table(box=box.ROUNDED, border_style="green")
    t.add_column("Metric", style="bold")
    t.add_column("Base Model", style="red")
    t.add_column("Fine-tuned", style="green")
    t.add_column("Delta", style="cyan")

    def fmt(v, suffix=""):
        return f"{v:.1f}{suffix}" if v is not None else "—"

    def delta(a, b, invert=False):
        if a is None or b is None:
            return "—"
        d = b - a
        sign = "+" if d > 0 else ""
        color = "green" if (d > 0) != invert else "red"
        return f"[{color}]{sign}{d:.1f}[/{color}]"

    base, ft = metrics[0], metrics[1]
    t.add_row("Greek Q&A Score (%)",
              fmt(base["qa"], "%"), fmt(ft["qa"], "%"),
              delta(base["qa"], ft["qa"]))
    t.add_row("Word Error Rate (%)",
              fmt(base["wer"], "%"), fmt(ft["wer"], "%"),
              delta(base["wer"], ft["wer"], invert=True))
    t.add_row("Char Error Rate (%)",
              fmt(base["cer"], "%"), fmt(ft["cer"], "%"),
              delta(base["cer"], ft["cer"], invert=True))

    console.print(t)
    input("\nPress Enter to return to menu...")


def show_logs():
    console.clear()
    logs_dir = BASE / "logs"
    logs = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not logs:
        console.print("[yellow]No log files found.[/yellow]")
        input("\nPress Enter...")
        return

    t = Table(box=box.SIMPLE, show_header=True)
    t.add_column("#", width=4)
    t.add_column("Log File", min_width=45)
    t.add_column("Size", width=10)

    for i, log in enumerate(logs[:20], 1):
        size = f"{log.stat().st_size / 1024:.1f} KB"
        t.add_row(str(i), log.name, size)

    console.print(t)
    choice = console.input("\nEnter number to view (or Enter to go back): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(logs):
            lines = logs[idx].read_text(encoding="utf-8", errors="replace").splitlines()
            console.print(Rule(logs[idx].name))
            # Show last 100 lines
            for line in lines[-100:]:
                console.print(line)
            input("\nPress Enter to return...")


def open_colab_notebook(relative_path: str):
    """Open a local .ipynb file in the default browser (Jupyter / VS Code)."""
    nb_path = BASE / relative_path
    if not nb_path.exists():
        console.print(f"[red]Notebook not found:[/red] {nb_path}")
        input("\nPress Enter to return...")
        return
    console.print(f"\n[yellow]Opening local notebook:[/yellow] {nb_path}")
    console.print("\n[cyan]To run on Colab:[/cyan]")
    console.print("  1. Upload the notebook to Google Colab (File → Upload notebook)")
    console.print("  2. Upload data/train_qa.jsonl and data/val_qa.jsonl to Google Drive/Gemma4GR/data/")
    console.print("  3. Select GPU runtime: Runtime → Change runtime type → L4 GPU")
    console.print("  4. Add HF_TOKEN in Colab Secrets (🔑 icon)")
    console.print(f"\n  Notebook path: {nb_path}\n")
    import webbrowser
    webbrowser.open(nb_path.as_uri())
    input("\nPress Enter to return to menu...")


def open_colab(url: str):
    console.print(f"\n[yellow]Opening Colab notebook in browser...[/yellow]")
    console.print(f"[dim]{url}[/dim]\n")
    console.print("[cyan]Remember:[/cyan]")
    console.print("  1. Select GPU runtime: Runtime → Change runtime type → GPU (L4 for E4B)")
    console.print("  2. Add HF_TOKEN in Colab Secrets (🔑 icon)")
    console.print("  3. Mount Google Drive for data access")
    console.print("  4. Your data is at: Google Drive → Gemma4GR/data/\n")
    webbrowser.open(url)
    input("\nPress Enter to return to menu...")


def main():
    if not (BASE / ".env").exists():
        console.print(Panel(
            "[yellow]No .env file found.[/yellow]\n"
            "Copy [cyan].env.example[/cyan] → [cyan].env[/cyan] and fill in your HF_TOKEN.",
            title="First Run Setup", border_style="yellow"
        ))
        input("\nPress Enter to continue...")

    dispatch = {
        # Phase 1 — STT
        "1":  lambda: run_script("training/setup_assets.py",          "setup"),
        "1P": lambda: run_script("training/setup_piper_sources.py",  "setup_piper_sources"),
        "2":  lambda: run_script("training/download_moira.py",         "download_moira"),
        "3":  lambda: run_script("training/generate_pairs.py",         "generate_pairs"),
        "4":  lambda: run_script("training/prepare_stt_dataset.py",    "prepare_stt"),
        "5":  lambda: run_script("training/prepare_piper_dataset.py",  "prepare_piper"),
        "6":  lambda: run_script("tests/run_before.py",                "test_before"),
        "7":  lambda: run_script("training/train_e2b_local.py",        "train_e2b"),
        "9":  lambda: run_script("training/train_piper.py",            "train_piper"),
        # Phase 2 — Q&A text
        "D":  lambda: run_script("training/generate_qa_pipeline.py",   "generate_qa"),
        "DD": lambda: run_script("training/run_phase2_auto.py",        "phase2_auto"),
        "E":  lambda: run_script("training/prepare_qa_dataset.py",     "prepare_qa"),
        "F":  lambda: run_script("training/train_qa_local.py",         "train_qa"),
        "G":  lambda: run_script("training/merge_adapters.py",         "merge"),
        # Colab
        "8A": lambda: open_colab(COLAB_E2B),
        "8B": lambda: open_colab(COLAB_E4B),
        "8C": lambda: open_colab(COLAB_STUDIO),
        "8D": lambda: open_colab_notebook("training/colab_qa_e4b.ipynb"),
        # Utils
        "A":  lambda: run_script("tests/run_after.py",                 "test_after"),
        "B":  show_compare,
        "C":  show_logs,
        "S":  lambda: run_script("training/validate_samples.py",       "validate_samples"),
        "Q":  lambda: sys.exit(0),
    }

    while True:
        show_menu()
        try:
            choice = console.input("[cyan]Choice:[/cyan] ").strip().upper()
        except EOFError:
            break
        if choice in dispatch:
            dispatch[choice]()
        else:
            console.print("[red]Invalid choice.[/red]")
            import time; time.sleep(0.8)


if __name__ == "__main__":
    main()
