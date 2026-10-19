"""Download the QASPER validation split (parquet) from the Hugging Face hub.

QASPER (Dasigi et al., 2021) contains questions written by NLP practitioners
about NLP papers, with gold evidence paragraphs and answers. Licence: CC BY 4.0.

Usage:  python scripts/download_qasper.py [--split validation] [--out data/qasper]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

URL = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out", type=Path, default=Path("data/qasper"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.split}.parquet"
    if dest.exists():
        print(f"{dest} already exists")
        return
    with httpx.stream("GET", URL.format(split=args.split), follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
