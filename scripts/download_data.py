"""Download and verify the official CC BY 4.0 UCI SECOM data."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
# SHA256s independently published from raw UCI source files.
CHECKSUMS = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
    "secom.names": "6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, sha in CHECKSUMS.items():
        path = RAW / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            print(f"[verified] {name}")
            continue
        try:
            print(f"[downloading] {name}")
            req = Request(BASE + name, headers={"User-Agent": "FabPulse-Research/1.0"})
            with urlopen(req, timeout=40) as resp:
                data = resp.read()
        except (URLError, OSError) as exc:
            print(f"Could not download {name}: {exc}", file=sys.stderr)
            print(f"Open {BASE + name} manually and save to {RAW}", file=sys.stderr)
            raise SystemExit(1) from exc
        received = hashlib.sha256(data).hexdigest()
        if received != sha:
            print(f"NOTICE: checksum differs for {name}. Expected {sha}, got {received}.", file=sys.stderr)
            print("Review this official source file before using it. File was NOT saved.", file=sys.stderr)
            raise SystemExit(1)
        path.write_bytes(data)
        print(f"[saved] {name} ({len(data):,} bytes)")
    print("Official UCI source files verified. CC BY 4.0 attribution required.")


if __name__ == "__main__":
    main()
