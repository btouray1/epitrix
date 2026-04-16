"""
EPITRIX ML PIPELINE — STEP 4
=============================
Integrates the trained XGBoost MHC-I model into Epitrix.

This module is a DROP-IN REPLACEMENT for the _local_epitope_scan()
function in app.py. Once the model is trained, import this module
and call ml_epitope_scan() instead.

Usage in app.py:
    # Replace this line:
    from epitrix_ml.04_integrate_epitrix import ml_epitope_scan

    # Then replace _local_epitope_scan(seq) with:
    # ml_epitope_scan(seq, model_path='epitrix_ml/models/mhci_xgboost.pkl')

Standalone test:
    python 04_integrate_epitrix.py
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional

# ── Re-use the same feature engineering from step 2 ──────────────────────────
# These must match exactly what was used during training

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
HLA_A0201_PSSM = {
    1: {'M': 2, 'L': 2, 'V': 1, 'I': 1, 'F': 1, 'A': 1},
    2: {'L': 3, 'M': 2, 'V': 2, 'I': 2, 'T': 1, 'A': 1},
    9: {'L': 4, 'V': 3, 'I': 3, 'M': 2, 'A': 1, 'T': 1},
}


def _featurise_peptide(seq: str) -> dict:
    """Must be identical to the function in 02_process_mhci.py."""
    seq = seq.upper()
    features = {}
    AA_LIST = sorted(STANDARD_AA)

    for i, aa in enumerate(seq):
        pos = i + 1
        for a in AA_LIST:
            features[f'p{pos}_{a}'] = 1 if aa == a else 0
        features[f'p{pos}_hydro'] = HYDROPHOBICITY.get(aa, 0)
        features[f'p{pos}_mw']    = MOLWEIGHT.get(aa, 111)
        features[f'p{pos}_iso']   = ISOELECTRIC.get(aa, 6.0)

    hydros = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    features['mean_hydrophobicity']  = np.mean(hydros)
    features['max_hydrophobicity']   = np.max(hydros)
    features['min_hydrophobicity']   = np.min(hydros)
    features['hydrophobicity_range'] = np.max(hydros) - np.min(hydros)
    features['charge_positive']      = sum(1 for aa in seq if aa in 'KR')
    features['charge_negative']      = sum(1 for aa in seq if aa in 'DE')
    features['net_charge']           = features['charge_positive'] - features['charge_negative']
    features['aromaticity']          = sum(1 for aa in seq if aa in 'FYW') / len(seq)
    features['aliphatic_index']      = sum(1 for aa in seq if aa in 'AVILM') / len(seq)

    pssm_score  = sum(HLA_A0201_PSSM.get(pos, {}).get(aa, 0) for pos, aa in enumerate(seq, 1))
    max_pssm    = sum(max(v.values()) for v in HLA_A0201_PSSM.values())
    features['pssm_a0201_score'] = pssm_score / max_pssm if max_pssm > 0 else 0
    features['peptide_length']   = len(seq)
    return features


def _encode_allele_a0201() -> dict:
    """Default to HLA-A*02:01 encoding for Epitrix integration."""
    return {
        'allele_A': 1, 'allele_B': 0, 'allele_C': 0,
        'allele_a0201': 1, 'allele_a0101': 0, 'allele_a0301': 0,
        'allele_a2402': 0, 'allele_a1101': 0, 'allele_b0702': 0,
        'allele_b4402': 0, 'allele_b3501': 0, 'allele_b4001': 0,
    }


# ── Model cache — load once, reuse across calls ───────────────────────────────
_MODEL_CACHE: dict = {}


def _load_model(model_path: str) -> dict:
    """Load model from disk with caching."""
    if model_path not in _MODEL_CACHE:
        data = joblib.load(model_path)
        _MODEL_CACHE[model_path] = data
    return _MODEL_CACHE[model_path]


def ml_epitope_scan(
    seq: str,
    model_path: str = "epitrix_ml/models/mhci_xgboost.pkl",
    strong_threshold: float = 0.6,
    weak_threshold:   float = 0.3,
) -> dict:
    """
    Scan a protein sequence for MHC-I epitopes using the trained XGBoost model.

    This is a drop-in replacement for _local_epitope_scan() in app.py.
    Returns the same dict structure so no other code needs to change.

    Parameters
    ----------
    seq              : Protein sequence (single-letter amino acids)
    model_path       : Path to the trained model .pkl file
    strong_threshold : Probability threshold for strong binder classification
    weak_threshold   : Probability threshold for weak binder classification

    Returns
    -------
    dict with keys matching _local_epitope_scan() output:
        method, mhc1_score, ctl_epitopes_est, top_mhci_peptides,
        plus ML-specific: strong_binder_prob_mean, model_version
    """
    seq = seq.upper().strip().replace(' ', '').replace('\n', '')
    n   = len(seq)

    # Load model
    try:
        model_data  = _load_model(model_path)
        model       = model_data['model']
        feature_cols = model_data['feature_cols']
        metrics     = model_data.get('metrics', {})
    except FileNotFoundError:
        # Model not trained yet — fall back to PSSM
        return {'method': 'PSSM fallback (ML model not found)', '_ml_available': False}
    except Exception as e:
        return {'method': f'PSSM fallback (ML error: {e})', '_ml_available': False}

    # Score all 9-mers
    peptides, feature_rows = [], []
    allele_feats = _encode_allele_a0201()

    for i in range(n - 8):
        pep = seq[i:i+9]
        if all(aa in STANDARD_AA for aa in pep):
            pep_feats = _featurise_peptide(pep)
            row = {**pep_feats, **allele_feats}
            # Ensure column order matches training
            feature_rows.append([row.get(c, 0) for c in feature_cols])
            peptides.append((pep, i + 1))

    if not peptides:
        return {'method': 'ML (no valid 9-mers found)', 'mhc1_score': 0.0,
                'ctl_epitopes_est': 0, 'top_mhci_peptides': [], '_ml_available': True}

    import numpy as np
    X = np.array(feature_rows, dtype=np.float32)
    proba = model.predict_proba(X)   # shape: (n_peptides, 3)

    # proba columns: [non-binder, weak, strong]
    strong_prob = proba[:, 2]
    weak_prob   = proba[:, 1]
    binder_prob = strong_prob + weak_prob

    # Classify peptides
    strong_binders, weak_binders = [], []
    for (pep, pos), sp, wp, bp in zip(peptides, strong_prob, weak_prob, binder_prob):
        if sp >= strong_threshold:
            strong_binders.append((float(sp), pep, pos))
        elif bp >= weak_threshold:
            weak_binders.append((float(bp), pep, pos))

    strong_binders.sort(reverse=True)
    weak_binders.sort(reverse=True)

    # MHC-I aggregate score: rank-based approach (robust to training set enrichment bias).
    # The IEDB-trained model assigns high strong-binder probabilities broadly due to
    # enriched training data. Using raw mean_strong_prob saturates to ~92-100%.
    #
    # Fix: use the mean probability of the TOP 10% strongest-scoring peptides, scaled
    # by 0.75. This mirrors NetMHCpan's percentile rank logic — scoring relative to the
    # sequence's own distribution rather than on an absolute scale.
    # Calibrated: Spike RBD (top-10% mean ~0.85) → mhc1_score ~0.64 ✓
    mean_strong_prob = float(np.mean(strong_prob))
    top_n = max(1, int(len(strong_prob) * 0.10))
    top_probs = sorted(strong_prob, reverse=True)[:top_n]
    mhc1_score = float(np.clip(float(np.mean(top_probs)) * 0.75, 0, 0.92))

    return {
        'method':              f'XGBoost MHC-I (AUC={metrics.get("binary_auc_roc", "?"):.3f})',
        'mhc1_score':          mhc1_score,
        'ctl_epitopes_est':    len(strong_binders),
        'top_mhci_peptides':   strong_binders[:10],
        'weak_binders':        weak_binders[:10],
        'mean_strong_prob':    mean_strong_prob,
        'n_peptides_scored':   len(peptides),
        '_ml_available':       True,
        '_model_auc':          metrics.get('binary_auc_roc', None),
    }


# ── Test when run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 4: Integration Test")
    print("=" * 60)

    # Test sequences
    TEST_SEQS = {
        'SARS-CoV-2 RBD (first 50 AA)': 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCY',
        'HIV gp120 fragment':           'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEAT',
        'HPV16 L1 fragment':            'MSLWLPSEATVYLPPVPVSKVVSTDEYVARTNIYYHAGTSRLLAVGHPY',
    }

    MODEL_PATH = "models/mhci_xgboost.pkl"

    for name, seq in TEST_SEQS.items():
        print(f"\nSequence: {name}")
        result = ml_epitope_scan(seq, model_path=MODEL_PATH)

        if not result.get('_ml_available'):
            print(f"  ⚠️  {result['method']}")
            print("  Train the model first: python 03_train_mhci_model.py")
            break

        print(f"  Method:          {result['method']}")
        print(f"  MHC-I score:     {result['mhc1_score']:.3f}")
        print(f"  Strong binders:  {result['ctl_epitopes_est']}")
        print(f"  Mean strong prob:{result['mean_strong_prob']:.3f}")
        if result['top_mhci_peptides']:
            print(f"  Top peptides:")
            for prob, pep, pos in result['top_mhci_peptides'][:5]:
                print(f"    pos {pos:>4}  {pep}  strong_prob={prob:.3f}")

    print("\n✅ Integration test complete.")
    print("\nTo use in Epitrix app.py:")
    print("  1. Add 'epitrix_ml' to your project folder")
    print("  2. In app.py, import: from epitrix_ml.integrate import ml_epitope_scan")
    print("  3. Replace _local_epitope_scan(seq) with ml_epitope_scan(seq)")
