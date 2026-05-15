"""Build a simple HTML+SVG loss chart from Piper CSVLogger metrics.csv (no matplotlib)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT_HTML = BASE / "output" / "piper_voice" / "metrics_curve.html"


def _find_metrics_csv() -> Path:
    root = BASE / "output" / "piper_voice" / "csv_metrics"
    if not root.is_dir():
        print(f"[ERROR] {root} not found.")
        sys.exit(1)
    cands = list(root.rglob("metrics.csv"))
    if not cands:
        print(f"[ERROR] No metrics.csv under {root}")
        sys.exit(1)
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> None:
    src = _find_metrics_csv()
    rows = _load_rows(src)
    gen_pts: list[list[float]] = []
    val_pts: list[list[float]] = []
    for row in rows:
        ep = row.get("epoch", "")
        if ep == "":
            continue
        e = float(ep)
        if row.get("loss_gen_all", "").strip():
            gen_pts.append([e, float(row["loss_gen_all"])])
        if row.get("val_loss", "").strip():
            val_pts.append([e, float(row["val_loss"])])
    if not gen_pts and not val_pts:
        print("[ERROR] No numeric rows to plot.")
        sys.exit(1)

    all_y = [p[1] for p in gen_pts] + [p[1] for p in val_pts]
    all_x = [p[0] for p in gen_pts] + [p[0] for p in val_pts]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if ymin <= ymax * 0.99:
        ymin = ymin - (ymax - ymin) * 0.05
    ymax = ymax + (ymax - max(min(all_y), 1e-6)) * 0.08 or ymax + 1

    W, H, pad = 880, 420, 56

    def sx(x: float) -> float:
        if xmax <= xmin:
            return pad + (W - 2 * pad) / 2
        return pad + (x - xmin) / (xmax - xmin) * (W - 2 * pad)

    def sy(y: float) -> float:
        return H - pad - (y - ymin) / (ymax - ymin) * (H - 2 * pad)

    def polyline(pts: list[list[float]], color: str) -> str:
        if len(pts) < 2:
            return ""
        d = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{d}"/>'

    def circles(pts: list[list[float]], color: str) -> str:
        return "".join(
            f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="4" fill="{color}"/>'
            for p in pts
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Piper training metrics</title>
<style>
  body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 24px; background: #1a1a1e; color: #e8e8ec; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; }}
  .src {{ color: #888; font-size: 0.85rem; margin-bottom: 12px; word-break: break-all; }}
  svg {{ background: #242428; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Piper loss curves</h1>
<p class="src">Source: {src.relative_to(BASE)}</p>
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">
  <text x="{pad}" y="24" fill="#aaa" font-size="12">loss</text>
  <text x="{W//2}" y="{H-12}" fill="#aaa" font-size="12" text-anchor="middle">epoch</text>
  {polyline(gen_pts, "#6bb3ff")}
  {polyline(val_pts, "#7bdc90")}
  {circles(gen_pts, "#6bb3ff")}
  {circles(val_pts, "#7bdc90")}
</svg>
<p style="font-size:0.9rem;margin-top:12px">
  <span style="color:#6bb3ff">&#9632;</span> loss_gen_all
  &nbsp; &nbsp;
  <span style="color:#7bdc90">&#9632;</span> val_loss
</p>
<script type="application/json" id="raw">{json.dumps(rows)}</script>
</body>
</html>
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Wrote {OUT_HTML.relative_to(BASE)}")


if __name__ == "__main__":
    main()
