"""
Download Greek CSS10 single-speaker pack from Kaggle into data/css10_greek/.

Dataset: https://www.kaggle.com/datasets/bryanpark/greek-single-speaker-speech-dataset

Uses the Kaggle API to resolve the download URL, then streams the zip with
``requests`` so slow links can use a long/no read-timeout (the stock CLI often
hits urllib read timeouts on ~1 Mbps).

Credentials:
  - ~/.kaggle/kaggle.json, or
  - KAGGLE_USERNAME + KAGGLE_KEY in .env (written to kaggle.json if missing).

Timeouts (optional, .env):
  KAGGLE_CONNECT_TIMEOUT — seconds to establish TLS (default 180).
  KAGGLE_READ_TIMEOUT — max idle seconds between chunks while streaming; empty or
    ``none`` / ``inf`` = no limit (default), recommended for slow downloads.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

BASE = Path(__file__).resolve().parent.parent
DATASET = "bryanpark/greek-single-speaker-speech-dataset"
OUT_DIR = BASE / "data" / "css10_greek"
CHUNK = 1024 * 1024


def _ensure_kaggle_json() -> None:
    home_kaggle = Path.home() / ".kaggle"
    cred_path = home_kaggle / "kaggle.json"
    if cred_path.is_file():
        return
    load_dotenv(BASE / ".env")
    user = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    if not user or not key:
        print(
            "[ERROR] No Kaggle credentials.\n"
            "  Either save Kaggle's API token file as:\n"
            f"    {cred_path}\n"
            "  Or add to .env:\n"
            "    KAGGLE_USERNAME=your_kaggle_username\n"
            "    KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "  Create a token: https://www.kaggle.com/settings -> API -> Create New Token",
            file=sys.stderr,
        )
        sys.exit(2)
    home_kaggle.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps({"username": user, "key": key}),
        encoding="utf-8",
    )


def _timeouts() -> tuple[float, float | None]:
    raw_c = os.getenv("KAGGLE_CONNECT_TIMEOUT", "180").strip()
    connect = float(raw_c) if raw_c else 180.0
    raw_r = os.getenv("KAGGLE_READ_TIMEOUT", "").strip().lower()
    if raw_r in ("", "none", "inf", "unlimited"):
        read_t: float | None = None
    else:
        read_t = float(raw_r)
    return connect, read_t


def _stream_zip(url: str, outfile: Path, timeout: tuple[float, float | None]) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "gemma4gr-css10-download"}) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        n_total = int(total) if total else None
        bar = tqdm(total=n_total, unit="B", unit_scale=True, unit_divisor=1024) if tqdm else None
        got = 0
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()
        print(f"  Saved {got / (1024 * 1024):.1f} MiB -> {outfile}")


def main() -> None:
    load_dotenv(BASE / ".env")
    _ensure_kaggle_json()

    api = KaggleApi()
    api.authenticate()
    owner_slug, dataset_slug, dataset_version_number = api.split_dataset_string(DATASET)

    print(f"Dataset URL: https://www.kaggle.com/datasets/{owner_slug}/{dataset_slug}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / f"{dataset_slug}.zip"
    timeout = _timeouts()
    print(f"  Stream timeout: connect={timeout[0]}s, read={timeout[1]!s} (None = no limit between chunks)")

    with api.build_kaggle_client() as kaggle:
        req = ApiDownloadDatasetRequest()
        req.owner_slug = owner_slug
        req.dataset_slug = dataset_slug
        req.dataset_version_number = dataset_version_number
        meta = kaggle.datasets.dataset_api_client.download_dataset(req)
        url = getattr(meta, "url", "") or ""
        if not url:
            print("[ERROR] Could not resolve download URL from Kaggle API.", file=sys.stderr)
            sys.exit(4)
        meta.close()

    _stream_zip(url, zip_path, timeout)

    print("  Unzipping ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
    zip_path.unlink(missing_ok=True)
    print(f"Done. Data under {OUT_DIR}")


if __name__ == "__main__":
    main()
