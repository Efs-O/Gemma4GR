"""
Gemma4GR — Greek Language Fine-tuning Pipeline
Entry point: python menu.py
"""
import sys, os, json, subprocess, webbrowser
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console()

BASE = Path(__file__).parent

COLAB_E2B = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Audio.ipynb"
COLAB_E4B = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Audio.ipynb"
COLAB_STUDIO = "https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb"

STEPS = [
    ("1", "Setup Assets",               "training/setup_assets.py",        "assets/piper/piper.exe"),
    ("2", "Download Moira Model",        "training/download_moira.py",      "models/moira/config.json"),
    ("3", "Generate Greek Audio Pairs",  "training/generate_pairs.py",      "data/raw_audio"),
    ("4", "Prepare STT Dataset",         "training/prepare_stt_dataset.py", "data/train_stt.jsonl"),
    ("5", "Prepare Piper Dataset",       "training/prepare_piper_dataset.py","data/piper_dataset/metadata.csv"),
    ("6", "Benchmark BEFORE Training",   "tests/run_before.py",             "tests/benchmark_results/base_qa.json"),
    ("7", "Train E2B — Local GPU",       "training/train_e2b_local.py",     "output/e2b_greek_stt/lora_adapter"),
    ("9", "Train Piper Voice (Docker)",  "training/train_piper.py",         "output/piper_voice/el_GR-gemma4gr-medium.onnx"),
    ("A", "Benchmark AFTER Training",    "tests/run_after.py",              "tests/benchmark_results/finetuned_qa.json"),
    ("B", "Compare Before/After",        None,                              None),
    ("C", "View Logs",                   None,                              None),
]

COLAB_ITEMS = [
    ("8a", "Open E2B Audio Colab",   COLAB_E2B),
    ("8b", "Open E4B Audio Colab",   COLAB_E4B),
    ("8c", "Open Unsloth Studio",    COLAB_STUDIO),
]


def step_status(artifact: str | None) -> Text:
    if artifact is None:
        return Text("—", style="dim")
    if Path(BASE / artifact).exists():
        return Text("✓ Done", style="bold green")
    return Text("Pending", style="dim")


def show_menu():
    console.clear()
    console.print(Rule("[bold blue]Gemma4GR — Greek Language Fine-tuning[/bold blue]"))

    # Main steps table
    t = Table(box=box.ROUNDED, border_style="blue", show_header=True, header_style="bold cyan")
    t.add_column("Key", width=5)
    t.add_column("Step", min_width=35)
    t.add_column("Status", width=10)

    for key, label, _, artifact in STEPS:
        t.add_row(f"[cyan]{key}[/cyan]", label, step_status(artifact))

    console.print(t)

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

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
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
        "1":  lambda: run_script("training/setup_assets.py",         "setup"),
        "2":  lambda: run_script("training/download_moira.py",        "download_moira"),
        "3":  lambda: run_script("training/generate_pairs.py",        "generate_pairs"),
        "4":  lambda: run_script("training/prepare_stt_dataset.py",   "prepare_stt"),
        "5":  lambda: run_script("training/prepare_piper_dataset.py", "prepare_piper"),
        "6":  lambda: run_script("tests/run_before.py",               "test_before"),
        "7":  lambda: run_script("training/train_e2b_local.py",       "train_e2b"),
        "8A": lambda: open_colab(COLAB_E2B),
        "8B": lambda: open_colab(COLAB_E4B),
        "8C": lambda: open_colab(COLAB_STUDIO),
        "9":  lambda: run_script("training/train_piper.py",           "train_piper"),
        "A":  lambda: run_script("tests/run_after.py",                "test_after"),
        "B":  show_compare,
        "C":  show_logs,
        "Q":  lambda: sys.exit(0),
    }

    while True:
        show_menu()
        choice = console.input("[cyan]Choice:[/cyan] ").strip().upper()
        if choice in dispatch:
            dispatch[choice]()
        else:
            console.print("[red]Invalid choice.[/red]")
            import time; time.sleep(0.8)


if __name__ == "__main__":
    main()
