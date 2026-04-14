"""
EPITRIX ML PIPELINE — STEP 1
=============================
Downloads MHC-I and MHC-II binding affinity datasets from IEDB.
Run this once. Files are saved to data/raw/.

Usage:
    python 01_download_iedb.py

Requirements:
    pip install requests pandas
"""

import os
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── IEDB bulk download URLs ───────────────────────────────────────────────────
# These are the official IEDB bulk data files (free, no login required)
DOWNLOADS = {
    "mhc_ligand_full": {
        "url": "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_v3.zip",
        "description": "MHC ligand elution + binding affinity data (MHC-I and MHC-II)",
        "filename": "mhc_ligand_full.zip",
    },
    "tcell_full": {
        "url": "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip",
        "description": "T cell epitope assay data",
        "filename": "tcell_full.zip",
    },
    "bcell_full": {
        "url": "https://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3.zip",
        "description": "B cell epitope assay data",
        "filename": "bcell_full.zip",
    },
}

def download_file(name: str, info: dict) -> Path:
    """Download and extract a zip file from IEDB."""
    out_path = RAW_DIR / info["filename"]

    if out_path.exists():
        print(f"  ✅ Already downloaded: {info['filename']}")
        return out_path

    print(f"  ⬇️  Downloading {name}: {info['description']}")
    print(f"      URL: {info['url']}")

    try:
        response = requests.get(info["url"], timeout=120, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunks = []

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r      Progress: {pct:.1f}% ({downloaded/1e6:.1f} MB)", end="")

        print()
        data = b"".join(chunks)

        # Save zip
        with open(out_path, "wb") as f:
            f.write(data)

        # Extract
        print(f"      Extracting to {RAW_DIR}/...")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(RAW_DIR)

        print(f"  ✅ Saved: {out_path}")
        return out_path

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to download {name}: {e}")
        return None


def verify_downloads():
    """Print what was downloaded and file sizes."""
    print("\n📁 Downloaded files:")
    for f in sorted(RAW_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"   {f.name:<45} {size_mb:.1f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 1: IEDB Data Download")
    print("=" * 60)
    print(f"Saving to: {RAW_DIR.resolve()}\n")

    for name, info in DOWNLOADS.items():
        download_file(name, info)

    verify_downloads()

    print("\n✅ Download complete. Run 02_process_mhci.py next.")
