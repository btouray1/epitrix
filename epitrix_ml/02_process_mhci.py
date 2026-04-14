"""
EPITRIX ML PIPELINE — STEP 2
=============================
Processes the raw IEDB MHC-I ligand data into a clean, model-ready dataset.

What this does:
  1. Loads mhc_ligand_full.csv from IEDB
  2. Filters to MHC-I human alleles with quantitative binding measurements
  3. Engineers sequence features (amino acid physicochemistry, PSSM scores)
  4. Binarises labels: strong binder (percentile <0.5%), weak (0.5-2%), non-binder (>2%)
  5. Saves processed dataset to data/processed/mhci_dataset.parquet

Usage:
    python 02_process_mhci.py

Expected input:  data/raw/mhc_ligand_full.csv  (from step 1)
Expected output: data/processed/mhci_dataset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Amino acid physicochemical feature tables ─────────────────────────────────
# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Molecular weight of amino acid residues
MOLWEIGHT = {
    'A': 89,  'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'Q': 146, 'E': 147, 'G': 75,  'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}

# Isoelectric point contributions
ISOELECTRIC = {
    'A': 6.01, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
    'Q': 5.65, 'E':  3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74,  'M': 5.74, 'F': 5.48, 'P': 6.30,
    'S': 5.68, 'T': 5.60,  'W': 5.89, 'Y': 5.66, 'V': 5.97,
}

# HLA-A*02:01 PSSM anchor scores (positions 1-9, simplified)
# From SYFPEITHI / Rammensee et al. 1999 — same as in Epitrix PSSM scanner
HLA_A0201_PSSM = {
    1: {'M': 2, 'L': 2, 'V': 1, 'I': 1, 'F': 1, 'A': 1},
    2: {'L': 3, 'M': 2, 'V': 2, 'I': 2, 'T': 1, 'A': 1},
    9: {'L': 4, 'V': 3, 'I': 3, 'M': 2, 'A': 1, 'T': 1},  # critical C-terminal anchor
}

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def is_valid_peptide(seq: str, length: int = 9) -> bool:
    """Check peptide is the right length and contains only standard amino acids."""
    return (
        isinstance(seq, str)
        and len(seq) == length
        and all(aa in STANDARD_AA for aa in seq.upper())
    )


def featurise_peptide(seq: str) -> dict:
    """
    Convert a peptide sequence into a numeric feature vector.

    Features engineered:
      - Per-position amino acid identity (one-hot, 9 positions × 20 AA = 180 features)
      - Per-position hydrophobicity (9 features)
      - Per-position molecular weight (9 features)
      - Per-position isoelectric point (9 features)
      - Aggregate: mean/max/min hydrophobicity, charge count, aromaticity
      - HLA-A*02:01 PSSM anchor score
      - Peptide length (always 9 here, included for extensibility)
    """
    seq = seq.upper()
    features = {}

    AA_LIST = sorted(STANDARD_AA)

    # Per-position features
    for i, aa in enumerate(seq):
        pos = i + 1

        # One-hot encoding per position
        for a in AA_LIST:
            features[f'p{pos}_{a}'] = 1 if aa == a else 0

        # Physicochemical per position
        features[f'p{pos}_hydro']  = HYDROPHOBICITY.get(aa, 0)
        features[f'p{pos}_mw']     = MOLWEIGHT.get(aa, 111)
        features[f'p{pos}_iso']    = ISOELECTRIC.get(aa, 6.0)

    # Aggregate features
    hydros = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    features['mean_hydrophobicity'] = np.mean(hydros)
    features['max_hydrophobicity']  = np.max(hydros)
    features['min_hydrophobicity']  = np.min(hydros)
    features['hydrophobicity_range']= np.max(hydros) - np.min(hydros)

    features['charge_positive'] = sum(1 for aa in seq if aa in 'KR')
    features['charge_negative'] = sum(1 for aa in seq if aa in 'DE')
    features['net_charge']      = features['charge_positive'] - features['charge_negative']
    features['aromaticity']     = sum(1 for aa in seq if aa in 'FYW') / len(seq)
    features['aliphatic_index'] = sum(1 for aa in seq if aa in 'AVILM') / len(seq)

    # HLA-A*02:01 PSSM score
    pssm_score = sum(
        HLA_A0201_PSSM.get(pos, {}).get(aa, 0)
        for pos, aa in enumerate(seq, 1)
    )
    max_pssm = sum(max(v.values()) for v in HLA_A0201_PSSM.values())
    features['pssm_a0201_score'] = pssm_score / max_pssm if max_pssm > 0 else 0

    features['peptide_length'] = len(seq)

    return features


def encode_allele(allele: str) -> dict:
    """
    Encode MHC allele as numeric features.
    Handles both human HLA and mouse H-2 alleles.
    """
    allele_raw = str(allele).strip()
    allele_up  = allele_raw.upper()
    features   = {}

    # ── Species flag ──────────────────────────────────────────────────────────
    features['is_human'] = 1 if 'HLA'  in allele_up else 0
    features['is_mouse'] = 1 if 'H2-'  in allele_up or 'H-2' in allele_up else 0

    # ── Human HLA gene ────────────────────────────────────────────────────────
    features['allele_HLA_A'] = 1 if 'HLA-A' in allele_up else 0
    features['allele_HLA_B'] = 1 if 'HLA-B' in allele_up else 0
    features['allele_HLA_C'] = 1 if 'HLA-C' in allele_up else 0

    # Common human alleles — covers ~85% of global population
    human_alleles = {
        'A*02:01': 'a0201', 'A*01:01': 'a0101', 'A*03:01': 'a0301',
        'A*24:02': 'a2402', 'A*11:01': 'a1101', 'A*02:03': 'a0203',
        'A*02:06': 'a0206', 'A*23:01': 'a2301', 'A*26:01': 'a2601',
        'B*07:02': 'b0702', 'B*44:02': 'b4402', 'B*35:01': 'b3501',
        'B*40:01': 'b4001', 'B*44:03': 'b4403', 'B*51:01': 'b5101',
        'B*57:01': 'b5701', 'B*58:01': 'b5801', 'C*07:02': 'c0702',
    }
    for allele_key, feat_name in human_alleles.items():
        features[f'allele_{feat_name}'] = 1 if allele_key in allele_raw else 0

    # ── Mouse H-2 alleles ─────────────────────────────────────────────────────
    # Common inbred strain alleles used in vaccine research
    mouse_alleles = {
        'H2-Kb':  'h2kb',   # C57BL/6 — most common lab strain
        'H2-Db':  'h2db',   # C57BL/6
        'H2-Kd':  'h2kd',   # BALB/c
        'H2-Dd':  'h2dd',   # BALB/c
        'H2-Ld':  'h2ld',   # BALB/c
        'H2-Kk':  'h2kk',   # CBA/J
        'H2-Dk':  'h2dk',   # CBA/J
        'H2-Kq':  'h2kq',   # DBA/1
        'H2-Dq':  'h2dq',   # DBA/1
        'H2-Ks':  'h2ks',   # A/J
    }
    for allele_key, feat_name in mouse_alleles.items():
        features[f'allele_{feat_name}'] = 1 if allele_key in allele_raw else 0

    # ── Mouse PSSM anchor score (H-2Kb) ──────────────────────────────────────
    # H-2Kb anchors: position 5 (Y/F) and position 8 (L/V/I/M) — Rammensee 1999
    features['allele_is_h2kb'] = 1 if 'H2-Kb' in allele_raw or 'H-2Kb' in allele_raw else 0
    features['allele_is_h2db'] = 1 if 'H2-Db' in allele_raw or 'H-2Db' in allele_raw else 0

    return features


# ── Mouse H-2Kb PSSM (anchor positions) ──────────────────────────────────────
# H-2Kb: position 5 = Y/F (aromatic), position 8 = L/V/I/M (aliphatic)
# Derived from Rammensee et al. Immunogenetics 1999 and SYFPEITHI database
H2_KB_PSSM = {
    5: {'Y': 4, 'F': 3, 'W': 2, 'H': 1},
    8: {'L': 3, 'V': 3, 'I': 3, 'M': 2, 'A': 1},
}

# H-2Db: position 5 = N/D/E/Q (polar), position 9 = M/L/V (aliphatic)
H2_DB_PSSM = {
    5: {'N': 3, 'D': 2, 'E': 2, 'Q': 2, 'S': 1},
    9: {'M': 4, 'L': 3, 'V': 3, 'I': 2},
}


def score_mouse_pssm(seq: str) -> dict:
    """Score a 9-mer peptide against H-2Kb and H-2Db PSSMs."""
    seq = seq.upper()
    scores = {}

    # H-2Kb
    kb_score  = sum(H2_KB_PSSM.get(pos, {}).get(aa, 0) for pos, aa in enumerate(seq, 1))
    kb_max    = sum(max(v.values()) for v in H2_KB_PSSM.values())
    scores['pssm_h2kb_score'] = kb_score / kb_max if kb_max > 0 else 0

    # H-2Db
    db_score  = sum(H2_DB_PSSM.get(pos, {}).get(aa, 0) for pos, aa in enumerate(seq, 1))
    db_max    = sum(max(v.values()) for v in H2_DB_PSSM.values())
    scores['pssm_h2db_score'] = db_score / db_max if db_max > 0 else 0

    return scores


def load_and_filter_mhci(csv_path: Path, species: str = 'human') -> pd.DataFrame:
    """
    Load IEDB MHC ligand data and filter to usable MHC-I 9-mer rows.

    Parameters
    ----------
    csv_path : Path to mhc_ligand_full.csv
    species  : 'human' (HLA alleles) or 'mouse' (H-2 alleles)
    """
    print(f"  Loading {csv_path.name} [{species.upper()} alleles]...")

    # IEDB CSV has 2 header rows — skip the first
    df = pd.read_csv(csv_path, header=1, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # ── Confirmed IEDB v3 column names ────────────────────────────────────────
    peptide_col     = 'Name'
    allele_col      = 'Name.6'
    ic50_col        = 'Quantitative measurement'
    qualitative_col = 'Qualitative Measurement'
    class_col       = 'Class'
    species_col     = 'Name.2'

    for col, label in [(peptide_col, 'peptide'), (allele_col, 'allele'),
                       (ic50_col, 'IC50'), (class_col, 'class')]:
        if col not in df.columns:
            print(f"  ❌ Column '{col}' ({label}) not found.")
            return pd.DataFrame(), peptide_col, allele_col

    # ── Filter 1: MHC Class I only ────────────────────────────────────────────
    df = df[df[class_col].astype(str).str.strip() == 'I'].copy()
    print(f"  After Class I filter: {len(df):,}")

    # ── Filter 2: Species-specific alleles ───────────────────────────────────
    if species == 'human':
        allele_mask = df[allele_col].astype(str).str.contains(
            r'^HLA-', na=False, regex=True
        )
        print(f"  Filtering for human HLA alleles...")
    elif species == 'mouse':
        allele_mask = df[allele_col].astype(str).str.contains(
            r'H2-|H-2', na=False, regex=True
        )
        print(f"  Filtering for mouse H-2 alleles...")
    else:
        raise ValueError(f"species must be 'human' or 'mouse', got '{species}'")

    df = df[allele_mask].copy()
    print(f"  After {species} allele filter: {len(df):,}")

    if len(df) == 0:
        print(f"  ❌ No {species} alleles found.")
        return pd.DataFrame(), peptide_col, allele_col

    print(f"  Top alleles:")
    print(df[allele_col].value_counts().head(10).to_string())

    # ── Filter 3: 9-mer peptides only ─────────────────────────────────────────
    df['peptide_clean'] = df[peptide_col].astype(str).str.strip().str.upper()
    df = df[df['peptide_clean'].apply(lambda x: is_valid_peptide(x, 9))].copy()
    print(f"  After 9-mer filter: {len(df):,}")

    if len(df) == 0:
        print(f"  ❌ No valid 9-mer peptides found.")
        return pd.DataFrame(), peptide_col, allele_col

    # ── Create binding labels ─────────────────────────────────────────────────
    df['ic50'] = pd.to_numeric(df[ic50_col], errors='coerce')
    has_ic50   = df['ic50'].notna()
    print(f"  Rows with IC50: {has_ic50.sum():,} / {len(df):,}")

    df['label_3class'] = -1
    df.loc[has_ic50 & (df['ic50'] > 500),  'label_3class'] = 0
    df.loc[has_ic50 & (df['ic50'] <= 500), 'label_3class'] = 1
    df.loc[has_ic50 & (df['ic50'] <= 50),  'label_3class'] = 2

    if qualitative_col in df.columns:
        no_ic50 = df['label_3class'] == -1
        qual = df.loc[no_ic50, qualitative_col].astype(str).str.lower().str.strip()
        df.loc[no_ic50 & qual.str.contains('negative',             na=False), 'label_3class'] = 0
        df.loc[no_ic50 & qual.str.contains('positive-low|positive-intermediate',
                                            na=False, regex=True),            'label_3class'] = 1
        df.loc[no_ic50 & qual.str.contains('positive-high|positive$',
                                            na=False, regex=True),            'label_3class'] = 2

    df = df[df['label_3class'] >= 0].copy()
    df['label_binary'] = (df['label_3class'] > 0).astype(int)

    print(f"\n  Binding label distribution:")
    vc     = df['label_3class'].value_counts().sort_index()
    labels = {0: 'Non-binder', 1: 'Weak binder', 2: 'Strong binder'}
    for k, v in vc.items():
        print(f"    {k} ({labels[k]}): {v:,}  ({v/len(df)*100:.1f}%)")

    # ── Stratified sample ─────────────────────────────────────────────────────
    MAX_PER_CLASS = 100_000
    print(f"\n  Sampling up to {MAX_PER_CLASS:,} per class...")
    df = df.groupby('label_3class', group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42)
    ).reset_index(drop=True)
    print(f"  Final dataset: {len(df):,} rows")

    return df, peptide_col, allele_col


def build_feature_matrix(df: pd.DataFrame, peptide_col: str,
                          allele_col: str, species: str = 'human') -> pd.DataFrame:
    """Build X (features) and y (labels) arrays."""

    print(f"\n  Building feature matrix [{species}]...")
    print(f"  Processing {len(df):,} peptides...")

    records = []
    skipped = 0
    total   = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 10000 == 0:
            print(f"\r  Progress: {idx:,}/{total:,} ({idx/total*100:.1f}%)", end="")
        try:
            pep    = str(row['peptide_clean'])
            allele = str(row.get(allele_col, 'HLA-A*02:01'))

            pep_feats    = featurise_peptide(pep)
            allele_feats = encode_allele(allele)
            mouse_feats  = score_mouse_pssm(pep)

            record = {**pep_feats, **allele_feats, **mouse_feats}

            ic50_val = pd.to_numeric(row.get('ic50', None), errors='coerce')
            record['log_ic50_available'] = 1 if not pd.isna(ic50_val) else 0
            record['log_ic50']           = float(np.log1p(ic50_val)) if not pd.isna(ic50_val) else 6.2

            record['label_binary'] = int(row.get('label_binary', 0))
            record['label_3class'] = int(row.get('label_3class', 0))
            record['peptide']      = pep
            record['allele']       = allele
            record['species']      = species
            records.append(record)

        except Exception:
            skipped += 1
            continue

    print(f"\r  Featurised: {len(records):,} | Skipped: {skipped:,}")
    return pd.DataFrame(records)


def load_once_split_both(csv_path: Path) -> tuple:
    """
    Load the 8.75 GB CSV exactly ONCE using chunked reading.
    Splits rows into human (HLA) and mouse (H-2) on the fly.
    Returns two DataFrames — never holds the full file in memory.
    """
    CHUNK_SIZE  = 50_000
    MAX_PER_CLASS = 100_000

    peptide_col     = 'Name'
    allele_col      = 'Name.6'
    ic50_col        = 'Quantitative measurement'
    qualitative_col = 'Qualitative Measurement'
    class_col       = 'Class'

    human_chunks, mouse_chunks = [], []
    total_read = 0

    print(f"  Reading {csv_path.name} in chunks of {CHUNK_SIZE:,}...")
    print(f"  (Single pass — no double memory load)")

    reader = pd.read_csv(csv_path, header=1, low_memory=False,
                         chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        total_read += len(chunk)
        if i % 20 == 0:
            print(f"\r  Read {total_read:,} rows...", end="")

        # Filter to Class I 9-mers with valid peptides
        chunk = chunk[chunk[class_col].astype(str).str.strip() == 'I'].copy()
        if len(chunk) == 0:
            continue

        chunk['peptide_clean'] = chunk[peptide_col].astype(str).str.strip().str.upper()
        chunk = chunk[chunk['peptide_clean'].apply(lambda x: is_valid_peptide(x, 9))]
        if len(chunk) == 0:
            continue

        # Add labels
        chunk['ic50'] = pd.to_numeric(chunk[ic50_col], errors='coerce')
        chunk['label_3class'] = -1
        has_ic50 = chunk['ic50'].notna()
        chunk.loc[has_ic50 & (chunk['ic50'] > 500),  'label_3class'] = 0
        chunk.loc[has_ic50 & (chunk['ic50'] <= 500), 'label_3class'] = 1
        chunk.loc[has_ic50 & (chunk['ic50'] <= 50),  'label_3class'] = 2
        if qualitative_col in chunk.columns:
            no_ic50 = chunk['label_3class'] == -1
            qual = chunk.loc[no_ic50, qualitative_col].astype(str).str.lower().str.strip()
            chunk.loc[no_ic50 & qual.str.contains('negative', na=False), 'label_3class'] = 0
            chunk.loc[no_ic50 & qual.str.contains('positive-low|positive-intermediate',
                      na=False, regex=True), 'label_3class'] = 1
            chunk.loc[no_ic50 & qual.str.contains('positive-high|positive$',
                      na=False, regex=True), 'label_3class'] = 2
        chunk = chunk[chunk['label_3class'] >= 0].copy()
        if len(chunk) == 0:
            continue
        chunk['label_binary'] = (chunk['label_3class'] > 0).astype(int)

        # Split by species
        alleles = chunk[allele_col].astype(str)
        human_mask = alleles.str.contains(r'^HLA-', na=False, regex=True)
        mouse_mask = alleles.str.contains(r'H2-|H-2',   na=False, regex=True)

        if human_mask.any():
            human_chunks.append(chunk[human_mask].copy())
        if mouse_mask.any():
            mouse_chunks.append(chunk[mouse_mask].copy())

    print(f"\r  Total rows read: {total_read:,}")

    def assemble(chunks, species_name):
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        print(f"\n  {species_name}: {len(df):,} rows before sampling")
        print(f"  Top alleles:")
        print(df[allele_col].value_counts().head(8).to_string())
        print(f"  Label distribution (before sampling):")
        labels = {0:'Non-binder', 1:'Weak binder', 2:'Strong binder'}
        for k, v in df['label_3class'].value_counts().sort_index().items():
            print(f"    {k} ({labels[k]}): {v:,}")
        # Stratified sample
        df = df.groupby('label_3class', group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=42)
        ).reset_index(drop=True)
        print(f"  After sampling: {len(df):,}")
        return df

    human_df = assemble(human_chunks, 'HUMAN')
    mouse_df  = assemble(mouse_chunks,  'MOUSE')
    return human_df, mouse_df, allele_col


def main():
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 2: Process MHC-I Data")
    print("  SINGLE-PASS: human + mouse extracted in one read")
    print("=" * 60)

    csv_path = None
    for name in ["mhc_ligand_full.csv", "mhc_ligand_full_v3.csv"]:
        candidate = RAW_DIR / name
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        candidates = list(RAW_DIR.glob("mhc_ligand*.csv"))
        if candidates:
            csv_path = candidates[0]
    if csv_path is None:
        print(f"❌ Could not find MHC ligand CSV in {RAW_DIR}")
        return

    print(f"  Input: {csv_path}")
    print(f"  File size: {csv_path.stat().st_size/1e9:.2f} GB\n")

    # ── Load once, split both ─────────────────────────────────────────────────
    human_df, mouse_df, allele_col = load_once_split_both(csv_path)

    all_feature_dfs = []

    for df, species in [(human_df, 'human'), (mouse_df, 'mouse')]:
        if len(df) == 0:
            print(f"\n  ⚠️  No {species} data — skipping.")
            continue

        print(f"\n{'─'*50}")
        print(f"  Featurising {species.upper()} ({len(df):,} rows)...")
        print(f"{'─'*50}")

        feature_df = build_feature_matrix(df, 'Name', allele_col, species=species)

        if len(feature_df) == 0:
            print(f"  ⚠️  Featurisation failed for {species}.")
            continue

        out_path = PROCESSED_DIR / f"mhci_dataset_{species}.parquet"
        feature_df.to_parquet(out_path, index=False)
        n_feat = len([c for c in feature_df.columns
                      if c not in ('label_binary','label_3class','peptide','allele','species')])
        print(f"\n  ✅ Saved {species}: {out_path}")
        print(f"     Rows: {len(feature_df):,} | Features: {n_feat}")
        all_feature_dfs.append(feature_df)

    # Also save human parquet under legacy name so training script fallback works
    if all_feature_dfs:
        human_feat = next((f for f in all_feature_dfs if 'human' in f['species'].iloc[0]), None)
        if human_feat is not None:
            legacy_path = PROCESSED_DIR / "mhci_dataset.parquet"
            human_feat.to_parquet(legacy_path, index=False)

    # Combined
    if len(all_feature_dfs) == 2:
        combined = pd.concat(all_feature_dfs, ignore_index=True)
        combined_path = PROCESSED_DIR / "mhci_dataset_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\n  ✅ Saved combined: {combined_path}  ({len(combined):,} rows)")

    print(f"\n{'='*60}")
    print("✅ Processing complete.")
    print("   Run 03_train_mhci_model.py next.")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print("✅ Processing complete.")
    print("   Run 03_train_mhci_model.py next.")
    print("   It will train 3 models: human, mouse, and combined.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
