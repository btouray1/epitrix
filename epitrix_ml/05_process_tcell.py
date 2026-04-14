"""
EPITRIX ML PIPELINE — STEP 5
==============================
Processes IEDB T cell assay data for immunogenicity prediction.

What this does:
  1. Loads tcell_full_v3.csv from IEDB
  2. Filters to usable assays (in vivo immunisation + ex vivo readout)
  3. Engineers peptide features (same as MHC-I model for consistency)
  4. Encodes delivery system, route, MHC class, assay type
  5. Creates two targets:
       label_binary      — Positive (1) or Negative (0)
       response_freq     — % subjects who responded (0-100, where available)
  6. Saves to data/processed/tcell_dataset.parquet

Usage:
    python 05_process_tcell.py

Expected input:  data/raw/tcell_full_v3.csv
Expected output: data/processed/tcell_dataset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Reuse physicochemical feature tables from MHC-I model ─────────────────────
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

HYDROPHOBICITY = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
MOLWEIGHT = {
    'A': 89,  'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'Q': 146, 'E': 147, 'G': 75,  'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}
ISOELECTRIC = {
    'A': 6.01, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
    'Q': 5.65, 'E':  3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02,
    'L': 5.98, 'K': 9.74,  'M': 5.74, 'F': 5.48, 'P': 6.30,
    'S': 5.68, 'T': 5.60,  'W': 5.89, 'Y': 5.66, 'V': 5.97,
}


def featurise_peptide(seq: str) -> dict:
    """Encode peptide as physicochemical features.
    Works for any length (unlike MHC-I model which was 9-mer only).
    Uses sliding window normalisation for variable-length peptides.
    """
    seq = seq.upper()
    n   = len(seq)
    features = {}

    AA_LIST = sorted(STANDARD_AA)

    # Per-position features for first 15 positions (covers most T cell epitopes)
    # Pad shorter peptides with zeros; truncate longer ones
    MAX_LEN = 15
    padded = seq[:MAX_LEN].ljust(MAX_LEN, 'X')
    for i, aa in enumerate(padded):
        pos = i + 1
        for a in AA_LIST:
            features[f'p{pos}_{a}'] = 1 if aa == a else 0
        features[f'p{pos}_hydro'] = HYDROPHOBICITY.get(aa, 0)
        features[f'p{pos}_mw']    = MOLWEIGHT.get(aa, 111)
        features[f'p{pos}_iso']   = ISOELECTRIC.get(aa, 6.0)

    # Aggregate features
    hydros = [HYDROPHOBICITY.get(aa, 0) for aa in seq if aa in STANDARD_AA]
    if hydros:
        features['mean_hydrophobicity']  = float(np.mean(hydros))
        features['max_hydrophobicity']   = float(np.max(hydros))
        features['min_hydrophobicity']   = float(np.min(hydros))
        features['hydrophobicity_range'] = float(np.max(hydros) - np.min(hydros))
    else:
        features['mean_hydrophobicity']  = 0.0
        features['max_hydrophobicity']   = 0.0
        features['min_hydrophobicity']   = 0.0
        features['hydrophobicity_range'] = 0.0

    features['peptide_length']   = n
    features['charge_positive']  = sum(1 for aa in seq if aa in 'KR')
    features['charge_negative']  = sum(1 for aa in seq if aa in 'DE')
    features['net_charge']       = features['charge_positive'] - features['charge_negative']
    features['aromaticity']      = sum(1 for aa in seq if aa in 'FYW') / max(n, 1)
    features['aliphatic_index']  = sum(1 for aa in seq if aa in 'AVILM') / max(n, 1)
    features['instability_proxy']= sum(1 for aa in seq if aa in 'DEGHKMNPQRST') / max(n, 1)

    return features


def encode_allele(allele: str) -> dict:
    """Encode MHC allele — human HLA and mouse H-2."""
    allele_raw = str(allele).strip()
    allele_up  = allele_raw.upper()
    features   = {}

    features['is_human'] = 1 if 'HLA' in allele_up else 0
    features['is_mouse'] = 1 if (
        'H2-'  in allele_up or
        'H-2'  in allele_up or
        'H2 '  in allele_up or     # catches "H2 class I", "H2 class II"
        allele_up.startswith('H2')  # catches "H2Kb" style
    ) else 0

    features['allele_HLA_A'] = 1 if 'HLA-A' in allele_up else 0
    features['allele_HLA_B'] = 1 if 'HLA-B' in allele_up else 0
    features['allele_HLA_C'] = 1 if 'HLA-C' in allele_up else 0
    features['allele_HLA_DR']= 1 if 'HLA-DR' in allele_up else 0
    features['allele_HLA_DQ']= 1 if 'HLA-DQ' in allele_up else 0
    features['allele_HLA_DP']= 1 if 'HLA-DP' in allele_up else 0

    human_alleles = {
        'A*02:01': 'a0201', 'A*01:01': 'a0101', 'A*03:01': 'a0301',
        'A*24:02': 'a2402', 'A*11:01': 'a1101', 'B*07:02': 'b0702',
        'B*44:02': 'b4402', 'B*57:01': 'b5701', 'B*35:01': 'b3501',
        'DRB1*01:01': 'dr0101', 'DRB1*03:01': 'dr0301',
        'DRB1*04:01': 'dr0401', 'DRB1*07:01': 'dr0701',
    }
    for k, v in human_alleles.items():
        features[f'allele_{v}'] = 1 if k in allele_raw else 0

    mouse_alleles = {
        'H2-Kb': 'h2kb', 'H2-Db': 'h2db', 'H2-Kd': 'h2kd',
        'H2-Dd': 'h2dd', 'H2-Ld': 'h2ld', 'H2-IAb': 'h2iab',
        'H2-IAd': 'h2iad', 'H2-IEd': 'h2ied', 'H2-Kk': 'h2kk',
    }
    for k, v in mouse_alleles.items():
        features[f'allele_{v}'] = 1 if k in allele_raw else 0

    return features


def encode_delivery(adjuvant: str, route: str, dose_schedule: str,
                     comments: str) -> dict:
    """
    Encode delivery system features from adjuvant, route, dose, comments fields.

    Delivery categories (mutually exclusive, in priority order):
      lipid_nanoparticle  — LNP, ionisable lipid, SM-102, ALC-0315 etc.
      liposome            — liposome, lipoplex, cationic lipid
      viral_vector        — vaccinia, adenovirus, AAV, lentivirus
      dna_vaccine         — plasmid DNA, pDNA
      mrna                — mRNA, saRNA
      protein_adjuvanted  — protein + alum / Freund's / MF59
      peptide_only        — peptide in saline, no adjuvant
      other_adjuvant      — other adjuvanted
    """
    adj = str(adjuvant).lower()  if pd.notna(adjuvant)     else ''
    rte = str(route).lower()     if pd.notna(route)        else ''
    dos = str(dose_schedule).lower() if pd.notna(dose_schedule) else ''
    cmt = str(comments).lower()  if pd.notna(comments)     else ''
    combined = adj + ' ' + dos + ' ' + cmt

    features = {}

    # Delivery category flags
    features['delivery_lnp'] = int(
        any(t in combined for t in [
            'lnp', 'lipid nanoparticle', 'ionizable', 'ionisable',
            'sm-102', 'alc-0315', 'mc3', 'dlin', 'dotap', 'dotma',
            'lipid nanopart'
        ])
    )
    features['delivery_liposome'] = int(
        any(t in combined for t in [
            'liposome', 'lipoplex', 'cationic lipid', 'lipid vesicle',
            'virosomes', 'virosome'
        ]) and not features['delivery_lnp']
    )
    features['delivery_viral'] = int(
        any(t in combined for t in [
            'vaccinia', 'adenovirus', 'adeno', 'aav ', 'lentivirus',
            'retrovirus', 'mva ', 'modified vaccinia', 'viral vector',
            'poxvirus', 'baculovirus', 'alphavirus', 'vvdd', 'recombinant virus'
        ])
    )
    features['delivery_dna'] = int(
        any(t in combined for t in [
            'plasmid', 'pdna', 'dna vaccine', 'dna construct',
            'dna immunization', ' dna '
        ]) and not features['delivery_viral']
    )
    features['delivery_mrna'] = int(
        any(t in combined for t in [
            'mrna', 'messenger rna', 'sarna', 'self-amplifying rna',
            'replicon', 'self amplifying'
        ])
    )
    features['delivery_protein'] = int(
        any(t in combined for t in [
            'freund', 'alum', 'mf59', 'as01', 'as02', 'as03', 'as04',
            'iscoms', 'iscom', 'qs-21', 'mpla', 'montanide', 'protein',
            'recombinant protein', 'subunit'
        ]) and not any([features['delivery_lnp'], features['delivery_liposome'],
                        features['delivery_viral'], features['delivery_dna'],
                        features['delivery_mrna']])
    )
    features['delivery_peptide_only'] = int(
        not any([features['delivery_lnp'], features['delivery_liposome'],
                 features['delivery_viral'], features['delivery_dna'],
                 features['delivery_mrna'], features['delivery_protein']])
        and any(t in combined for t in ['peptide', 'saline', 'pbs', 'no adjuvant'])
    )
    features['delivery_other'] = int(
        not any(v for k, v in features.items() if k.startswith('delivery_'))
    )

    # Lipid-specific adjuvant sub-types (for lipid category rows)
    features['adjuvant_freunds']   = int('freund' in combined)
    features['adjuvant_alum']      = int('alum' in combined or 'alumin' in combined)
    features['adjuvant_mf59']      = int('mf59' in combined or 'squalene' in combined)
    features['adjuvant_tlr']       = int(
        any(t in combined for t in ['cpg', 'poly i:c', 'poly(i:c)', 'mpla',
                                     'tlr', 'imiquimod', 'r848', 'poly-iclc'])
    )
    features['adjuvant_none']      = int(
        any(t in combined for t in ['no adjuvant', 'saline', 'pbs only', 'vehicle'])
    )

    # Route encoding
    routes = {
        'intramuscular':   'route_im',
        'intraperitoneal': 'route_ip',
        'subcutaneous':    'route_sc',
        'intranasal':      'route_in',
        'intravenous':     'route_iv',
        'intradermal':     'route_id',
        'oral':            'route_oral',
    }
    for route_name, feat_name in routes.items():
        features[feat_name] = int(route_name in rte)

    # Dose information (log-transform if numeric available)
    dose_num = None
    import re
    match = re.search(r'(\d+\.?\d*)\s*(µg|ug|mg|ng)', dos + cmt)
    if match:
        val  = float(match.group(1))
        unit = match.group(2).lower()
        if unit in ('mg',):
            val *= 1000
        elif unit in ('ng',):
            val /= 1000
        dose_num = val
    features['log_dose_ug']           = float(np.log1p(dose_num)) if dose_num else 0.0
    features['dose_available']        = int(dose_num is not None)

    # Number of doses
    n_doses = 1
    match2 = re.search(r'(\d+)\s*dose', dos + cmt)
    if match2:
        n_doses = int(match2.group(1))
    features['n_doses']               = min(n_doses, 10)

    return features


def encode_assay(method: str, response_measured: str, cell_type: str) -> dict:
    """Encode assay type — important because different assays have different
    sensitivity and measure different aspects of T cell immunity."""
    meth = str(method).lower()          if pd.notna(method)           else ''
    resp = str(response_measured).lower() if pd.notna(response_measured) else ''
    cell = str(cell_type).lower()       if pd.notna(cell_type)        else ''

    features = {}

    # Assay sensitivity hierarchy (higher = more sensitive / quantitative)
    features['assay_elispot']       = int('elispot' in meth or 'elispot' in resp)
    features['assay_intracellular'] = int(
        any(t in meth for t in ['intracellular', 'ics', 'ifn-g', 'ifn-γ']))
    features['assay_tetramer']      = int(
        any(t in meth for t in ['tetramer', 'multimer', 'dextramer']))
    features['assay_proliferation'] = int('proliferat' in meth or 'thymidine' in meth)
    features['assay_cytotoxicity']  = int(
        any(t in meth for t in ['cytotox', 'ctl', 'killing', 'cr51', 'cfse']))
    features['assay_elisa']         = int('elisa' in meth)
    features['assay_other']         = int(
        not any(v for k, v in features.items() if k.startswith('assay_')))

    # T cell subset
    features['tcell_cd8']   = int(
        'cd8' in cell or 'ctl' in cell or 'cytotox' in cell
        or ('class' in resp and 'i' in resp and 'ii' not in resp))
    features['tcell_cd4']   = int(
        'cd4' in cell or 'helper' in cell or 'th1' in cell or 'th2' in cell
        or 'proliferat' in resp)
    features['tcell_mixed'] = int(not features['tcell_cd8'] and not features['tcell_cd4'])

    # Cytokine measured
    features['cytokine_ifng']   = int(
        any(t in meth + resp for t in ['ifn-g', 'ifn-γ', 'ifng', 'interferon']))
    features['cytokine_il2']    = int('il-2' in meth + resp or 'il2' in meth + resp)
    features['cytokine_tnfa']   = int('tnf' in meth + resp)
    features['cytokine_multi']  = int(
        sum([features['cytokine_ifng'], features['cytokine_il2'],
             features['cytokine_tnfa']]) > 1)

    return features


def load_and_filter_tcell(csv_path: Path) -> pd.DataFrame:
    """Load and filter T cell data to usable immunisation assays."""
    print(f"  Loading {csv_path.name} in chunks...")

    CHUNK_SIZE = 50_000
    kept_chunks = []
    total_read  = 0

    reader = pd.read_csv(csv_path, header=1, low_memory=False,
                         chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        total_read += len(chunk)
        if i % 10 == 0:
            print(f"\r  Read {total_read:,} rows...", end="")

        # ── Filter 1: Must be in vivo immunisation (not in vitro stimulation only)
        if 'Process Type' in chunk.columns:
            chunk = chunk[
                chunk['Process Type'].astype(str).str.contains(
                    'Administration in vivo|immunization|immunisation',
                    na=False, case=False, regex=True
                )
            ].copy()

        if len(chunk) == 0:
            continue

        # ── Filter 2: Must have a peptide sequence in Name column
        if 'Name' not in chunk.columns:
            continue
        chunk['peptide_clean'] = chunk['Name'].astype(str).str.strip().str.upper()
        chunk = chunk[
            chunk['peptide_clean'].apply(
                lambda x: 4 <= len(x) <= 30
                and all(c in STANDARD_AA for c in x)
            )
        ].copy()

        if len(chunk) == 0:
            continue

        # ── Filter 3: Must have a binding outcome
        if 'Qualitative Measurement' not in chunk.columns:
            continue
        chunk = chunk[chunk['Qualitative Measurement'].notna()].copy()

        if len(chunk) > 0:
            kept_chunks.append(chunk)

    print(f"\r  Total read: {total_read:,}")
    if not kept_chunks:
        return pd.DataFrame()

    df = pd.concat(kept_chunks, ignore_index=True)
    print(f"  After filters: {len(df):,} rows")
    return df


def build_tcell_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full feature matrix from filtered T cell dataframe."""

    print(f"\n  Building T cell feature matrix...")
    print(f"  Processing {len(df):,} assays...")

    # Identify column names from actual data
    allele_col   = 'Name.10'
    method_col   = 'Method'
    response_col = 'Response measured'
    qual_col     = 'Qualitative Measurement'
    freq_col     = 'Response Frequency (%)'
    adj_col      = 'Adjuvants'
    route_col    = 'Route'
    dose_col     = 'Dose Schedule'
    class_col    = 'Class'
    comments_col = 'Comments.1'
    cell_col     = 'Name.8'
    n_tested_col = 'Number of Subjects Tested'
    n_pos_col    = 'Number of Subjects Positive'

    records = []
    skipped = 0
    total   = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 10000 == 0:
            print(f"\r  Progress: {idx:,}/{total:,} ({idx/total*100:.1f}%)", end="")
        try:
            pep    = str(row['peptide_clean'])
            allele = str(row.get(allele_col, '')) if allele_col in df.columns else ''

            pep_feats      = featurise_peptide(pep)
            allele_feats   = encode_allele(allele)
            delivery_feats = encode_delivery(
                row.get(adj_col),
                row.get(route_col),
                row.get(dose_col),
                row.get(comments_col)
            )
            assay_feats    = encode_assay(
                row.get(method_col),
                row.get(response_col),
                row.get(cell_col)
            )

            # ── MHC class ─────────────────────────────────────────────────────
            mhc_class = str(row.get(class_col, '')).strip()
            mhc_class_i  = int(mhc_class == 'I')
            mhc_class_ii = int(mhc_class == 'II')

            # ── Labels ────────────────────────────────────────────────────────
            qual = str(row.get(qual_col, '')).lower().strip()
            label_binary = 1 if 'positive' in qual else 0 if 'negative' in qual else None
            if label_binary is None:
                skipped += 1
                continue

            # Response frequency — use directly if available, else infer from
            # subjects tested/positive, else use binary label
            freq = pd.to_numeric(row.get(freq_col), errors='coerce')
            if pd.isna(freq):
                n_tested = pd.to_numeric(row.get(n_tested_col), errors='coerce')
                n_pos    = pd.to_numeric(row.get(n_pos_col),    errors='coerce')
                if not pd.isna(n_tested) and not pd.isna(n_pos) and n_tested > 0:
                    freq = (n_pos / n_tested) * 100
                else:
                    freq = 100.0 if label_binary == 1 else 0.0

            # Sample size as confidence weight (more subjects = more reliable)
            n_tested_val = pd.to_numeric(row.get(n_tested_col), errors='coerce')
            n_subjects   = float(n_tested_val) if not pd.isna(n_tested_val) else 1.0

            record = {
                **pep_feats, **allele_feats,
                **delivery_feats, **assay_feats,
                'mhc_class_i':   mhc_class_i,
                'mhc_class_ii':  mhc_class_ii,
                'label_binary':  label_binary,
                'response_freq': float(np.clip(freq, 0, 100)),
                'n_subjects':    min(n_subjects, 200),
                'peptide':       pep,
                'allele':        allele,
            }
            records.append(record)

        except Exception:
            skipped += 1
            continue

    print(f"\r  Featurised: {len(records):,} | Skipped: {skipped:,}")
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 5: Process T Cell Data")
    print("=" * 60)

    csv_path = RAW_DIR / "tcell_full_v3.csv"
    if not csv_path.exists():
        candidates = list(RAW_DIR.glob("tcell_full*.csv"))
        if candidates:
            csv_path = candidates[0]
        else:
            print(f"❌ tcell_full_v3.csv not found in {RAW_DIR}")
            return

    print(f"  Input: {csv_path}")
    print(f"  File size: {csv_path.stat().st_size/1e9:.2f} GB\n")

    # ── Load and filter ───────────────────────────────────────────────────────
    df = load_and_filter_tcell(csv_path)
    if len(df) == 0:
        print("❌ No data after filtering.")
        return

    # ── Delivery system summary ───────────────────────────────────────────────
    print(f"\n  Delivery system overview (Adjuvants column):")
    print(df['Adjuvants'].value_counts().head(15).to_string())
    print(f"\n  MHC class distribution:")
    print(df['Class'].value_counts().to_string())
    print(f"\n  Response frequency available: "
          f"{df['Response Frequency (%)'].notna().sum():,} / {len(df):,}")
    print(f"  Qualitative labels: "
          f"{df['Qualitative Measurement'].value_counts().to_string()}")

    # ── Build features ────────────────────────────────────────────────────────
    feature_df = build_tcell_features(df)
    if len(feature_df) == 0:
        print("❌ Feature matrix empty.")
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n  Dataset summary:")
    print(f"    Total assays:       {len(feature_df):,}")
    print(f"    Positive responses: {feature_df['label_binary'].sum():,} "
          f"({feature_df['label_binary'].mean()*100:.1f}%)")
    print(f"    Negative responses: {(feature_df['label_binary']==0).sum():,}")
    print(f"    Mean response freq: {feature_df['response_freq'].mean():.1f}%")
    print(f"    Features:           {len([c for c in feature_df.columns if c not in ('label_binary','response_freq','peptide','allele','n_subjects')]):,}")
    print(f"\n  Delivery breakdown:")
    for col in [c for c in feature_df.columns if c.startswith('delivery_')]:
        n = feature_df[col].sum()
        if n > 0:
            print(f"    {col:<30} {n:>6,}  ({n/len(feature_df)*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "tcell_dataset.parquet"
    feature_df.to_parquet(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path}")
    print(f"     Rows: {len(feature_df):,} | "
          f"Features: {len([c for c in feature_df.columns if c not in ('label_binary','response_freq','peptide','allele','n_subjects')]):,}")

    print(f"\n{'='*60}")
    print("✅ Processing complete.")
    print("   Run 06_train_tcell_model.py next.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
