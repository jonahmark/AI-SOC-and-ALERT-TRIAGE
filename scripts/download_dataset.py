"""Download the NSL-KDD dataset files into data/raw/.

Primary source: the UNB CIC NSL-KDD dataset page
(https://www.unb.ca/cic/datasets/nsl-kdd.html). This helper fetches the
two '+' files from a public GitHub mirror of that dataset and verifies
the expected record counts.

Usage: python scripts/download_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {
    "KDDTrain+.txt": 125_973,
    "KDDTest+.txt": 22_544,
}


def main() -> int:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected_rows in FILES.items():
        dest = config.DATA_RAW_DIR / name
        if dest.exists():
            n = sum(1 for _ in dest.open("rb"))
            if n == expected_rows:
                print(f"[skip] {name} already present ({n} records).")
                continue
            print(f"[warn] {name} exists but has {n} rows "
                  f"(expected {expected_rows}); re-downloading.")
        print(f"[get ] {name} ...")
        resp = requests.get(MIRROR + name.replace("+", "%2B"), timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        n = sum(1 for _ in dest.open("rb"))
        status = "OK" if n == expected_rows else "MISMATCH"
        print(f"[done] {name}: {n} records ({status})")
        if status == "MISMATCH":
            return 1
    print("Dataset ready in", config.DATA_RAW_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
