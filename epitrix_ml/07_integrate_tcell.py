"""
EPITRIX ML PIPELINE — STEP 7
==============================
Integrates trained T cell immunogenicity models into Epitrix.

Provides ml_tcell_scan() — a drop-in function for app.py that:
  1. Predicts T cell immunogenicity (Positive probability)
  2. Predicts response frequency % where reliable
  3. Applies LNP modulation from Epitrix's existing innate predictions
  4. Returns results in the same dict structure as the rest of Epitrix

Usage in app.py:
    from epitrix_ml.tcell_integrate import ml_tcell_predict

Standalone test:
    python 07_integrate_tcell.py
"""

import numpy as np
import joblib
from pathlib import Path

# ── Reuse feature engineering from 05_process_tcell.py ───────────────────────
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

MAX_LEN = 15


def _featurise_peptide(seq: str):
    """Peptide feature engineering — must match 05_process_tcell.py exactly."""
    seq = seq.upper()
    n   = len(seq)
    features = {}
    AA_LIST = sorted(STANDARD_AA)

    padded = seq[:MAX_LEN].ljust(MAX_LEN, 'X')
    for i, aa in enumerate(padded):
        pos = i + 1
        for a in AA_LIST:
            features[f'p{pos}_{a}'] = 1 if aa == a else 0
        features[f'p{pos}_hydro'] = HYDROPHOBICITY.get(aa, 0)
        features[f'p{pos}_mw']    = MOLWEIGHT.get(aa, 111)
        features[f'p{pos}_iso']   = ISOELECTRIC.get(aa, 6.0)

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

    features['peptide_length']    = n
    features['charge_positive']   = sum(1 for aa in seq if aa in 'KR')
    features['charge_negative']   = sum(1 for aa in seq if aa in 'DE')
    features['net_charge']        = features['charge_positive'] - features['charge_negative']
    features['aromaticity']       = sum(1 for aa in seq if aa in 'FYW') / max(n, 1)
    features['aliphatic_index']   = sum(1 for aa in seq if aa in 'AVILM') / max(n, 1)
    features['instability_proxy'] = sum(1 for aa in seq if aa in 'DEGHKMNPQRST') / max(n, 1)

    return features


def _default_allele_features(species: str):
    """Default allele encoding for scanning (assumes most common allele)."""
    features = {
        'is_human': 1 if species == 'human' else 0,
        'is_mouse': 1 if species == 'mouse' else 0,
        'allele_HLA_A': 1 if species == 'human' else 0,
        'allele_HLA_B': 0, 'allele_HLA_C': 0,
        'allele_HLA_DR': 0, 'allele_HLA_DQ': 0, 'allele_HLA_DP': 0,
        'allele_a0201': 1 if species == 'human' else 0,
        'allele_a0101': 0, 'allele_a0301': 0, 'allele_a2402': 0,
        'allele_a1101': 0, 'allele_b0702': 0, 'allele_b4402': 0,
        'allele_b5701': 0, 'allele_b3501': 0,
        'allele_dr0101': 0, 'allele_dr0301': 0,
        'allele_dr0401': 0, 'allele_dr0701': 0,
        'allele_h2kb': 1 if species == 'mouse' else 0,
        'allele_h2db': 0, 'allele_h2kd': 0, 'allele_h2dd': 0,
        'allele_h2ld': 0, 'allele_h2iab': 0, 'allele_h2iad': 0,
        'allele_h2ied': 0, 'allele_h2kk': 0,
    }
    return features


def _lnp_delivery_features():
    """Delivery features for LNP-delivered mRNA (Epitrix's primary use case)."""
    return {
        'delivery_lnp': 1, 'delivery_liposome': 0, 'delivery_viral': 0,
        'delivery_dna': 0, 'delivery_mrna': 1, 'delivery_protein': 0,
        'delivery_peptide_only': 0, 'delivery_other': 0,
        'adjuvant_freunds': 0, 'adjuvant_alum': 0, 'adjuvant_mf59': 0,
        'adjuvant_tlr': 0, 'adjuvant_none': 0,
        'route_im': 1, 'route_ip': 0, 'route_sc': 0,
        'route_in': 0, 'route_iv': 0, 'route_id': 0, 'route_oral': 0,
        'log_dose_ug': float(np.log1p(100)),  # typical 100 µg mRNA dose
        'dose_available': 1,
        'n_doses': 2,  # typical 2-dose prime-boost
    }


def _cd8_assay_features():
    """Default assay features for CD8+ T cell ELISPOT (most common readout)."""
    return {
        'assay_elispot': 1, 'assay_intracellular': 0, 'assay_tetramer': 0,
        'assay_proliferation': 0, 'assay_cytotoxicity': 0,
        'assay_elisa': 0, 'assay_other': 0,
        'tcell_cd8': 1, 'tcell_cd4': 0, 'tcell_mixed': 0,
        'cytokine_ifng': 1, 'cytokine_il2': 0,
        'cytokine_tnfa': 0, 'cytokine_multi': 0,
        'mhc_class_i': 1, 'mhc_class_ii': 0,
    }


# ── Model cache ───────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _load_model(path: str):
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def _find_model(name: str, base_dirs: list) :
    """Search for a model file across multiple possible locations."""
    import os
    for d in base_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def ml_tcell_predict(
    peptides: list,
    species:  str = 'human',
    innate_prediction=None,
    model_base_dir: str = 'epitrix_ml/models',
):
    """
    Predict T cell immunogenicity for a list of peptides using trained XGBoost models.

    Parameters
    ----------
    peptides          : list of peptide sequences (strings)
    species           : 'human' or 'mouse'
    innate_prediction : dict from Epitrix innate prediction module.
                        Used to apply LNP modulation. Keys used:
                          'TLR7_8', 'DC_maturation', 'antigen_expression'
                        If None, uses default LNP assumptions.
    model_base_dir    : path to directory containing .pkl model files

    Returns
    -------
    dict with:
        immunogenic_peptides  — list of (probability, peptide) tuples, sorted desc
        mean_immunogenicity   — mean positive probability across all peptides
        response_freq_pct     — predicted population response frequency (%)
        n_immunogenic         — count of peptides above 0.5 threshold
        method                — model name and AUC
        lnp_modulated         — whether LNP modulation was applied
        _available            — True if models loaded successfully
    """
    import os

    base_dirs = [
        model_base_dir,
        'models',
        os.path.join(os.path.dirname(__file__), 'models'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    ]

    # Find classifier
    clf_name = f"tcell_xgboost_classifier_{species}.pkl"
    clf_path = _find_model(clf_name, base_dirs)
    if clf_path is None:
        clf_path = _find_model("tcell_xgboost_classifier_combined.pkl", base_dirs)
    if clf_path is None:
        return {'_available': False, 'error': 'T cell model not found'}

    # Find regressor
    reg_path = _find_model("tcell_xgboost_regressor.pkl", base_dirs)

    try:
        clf_data = _load_model(clf_path)
        clf      = clf_data['model']
        feat_cols = clf_data['feature_cols']
        clf_auc   = clf_data['metrics'].get('auc_roc', 0)

        reg_data = _load_model(reg_path) if reg_path else None
        reg      = reg_data['model'] if reg_data else None

    except Exception as e:
        return {'_available': False, 'error': str(e)}

    # ── LNP modulation factor ─────────────────────────────────────────────────
    # Combines innate activation signals that are known to modulate T cell priming
    if innate_prediction:
        tlr    = float(innate_prediction.get('TLR7_8', 0.6))
        dc_mat = float(innate_prediction.get('DC_maturation', 0.6))
        ag_exp = float(innate_prediction.get('antigen_expression', 0.7))
        # Weighted combination — TLR activation and DC maturation are
        # the primary drivers of T cell priming efficiency
        lnp_factor = float(np.clip(
            0.4 * tlr + 0.35 * dc_mat + 0.25 * ag_exp, 0.2, 1.0
        ))
        lnp_modulated = True
    else:
        lnp_factor    = 0.65   # default mid-range LNP assumption
        lnp_modulated = False

    # ── Score all peptides ────────────────────────────────────────────────────
    allele_feats   = _default_allele_features(species)
    delivery_feats = _lnp_delivery_features()
    assay_feats    = _cd8_assay_features()

    feature_rows = []
    valid_peptides = []

    for pep in peptides:
        pep = str(pep).upper().strip()
        if len(pep) < 4 or len(pep) > 30:
            continue
        if not all(c in STANDARD_AA for c in pep):
            continue

        pep_feats = _featurise_peptide(pep)
        row = {**pep_feats, **allele_feats, **delivery_feats, **assay_feats}
        feature_rows.append([row.get(c, 0) for c in feat_cols])
        valid_peptides.append(pep)

    if not valid_peptides:
        return {
            '_available': True,
            'immunogenic_peptides': [],
            'mean_immunogenicity':  0.0,
            'response_freq_pct':    0.0,
            'n_immunogenic':        0,
            'method':               f'T cell XGBoost ({species})',
            'lnp_modulated':        lnp_modulated,
        }

    X = np.array(feature_rows, dtype=np.float32)

    # Classifier probabilities
    proba = clf.predict_proba(X)[:, 1]

    # Apply LNP modulation — scale probabilities toward the lnp_factor
    # Formula: final = base_prob × (0.5 + 0.5 × lnp_factor)
    # This means a strong LNP (factor=1.0) multiplies by 1.0 (no change)
    # A weak LNP (factor=0.2) multiplies by 0.6 (reduces immunogenicity)
    modulated_proba = proba * (0.5 + 0.5 * lnp_factor)
    modulated_proba = np.clip(modulated_proba, 0, 1)

    # Response frequency from regressor (back-transform arcsin-sqrt)
    if reg is not None:
        reg_pred     = reg.predict(X)
        response_pct = float(np.mean(
            np.clip(np.sin(reg_pred) ** 2 * 100, 0, 100)
        ))
        # Also modulate response frequency by LNP factor
        response_pct = float(np.clip(response_pct * (0.5 + 0.5 * lnp_factor), 0, 100))
    else:
        # Fall back to classifier probability × 100
        response_pct = float(np.mean(modulated_proba) * 100)

    # Rank peptides by immunogenicity probability
    ranked = sorted(
        zip(modulated_proba.tolist(), valid_peptides),
        key=lambda x: x[0],
        reverse=True
    )

    n_immunogenic = int((modulated_proba > 0.5).sum())

    return {
        '_available':           True,
        'immunogenic_peptides': ranked[:15],
        'mean_immunogenicity':  float(np.mean(modulated_proba)),
        'response_freq_pct':    response_pct,
        'n_immunogenic':        n_immunogenic,
        'n_peptides_scored':    len(valid_peptides),
        'lnp_factor':           float(lnp_factor),
        'lnp_modulated':        lnp_modulated,
        'method':               f'T cell XGBoost ({species}, AUC={clf_auc:.3f})',
        'clf_auc':              float(clf_auc),
        'species':              species,
    }


def scan_protein_sequence(
    sequence: str,
    species:  str = 'human',
    window:   int = 9,
    innate_prediction=None,
    model_base_dir: str = 'epitrix_ml/models',
):
    """
    Scan a full protein sequence for T cell immunogenic peptides.
    Slides a window across the sequence and scores all overlapping peptides.

    Parameters
    ----------
    sequence  : full protein sequence (single-letter amino acids)
    species   : 'human' or 'mouse'
    window    : peptide length to scan (default 9 for MHC-I; use 15 for MHC-II)
    innate_prediction : from Epitrix innate model for LNP modulation
    """
    seq = sequence.upper().strip().replace(' ', '').replace('\n', '')
    if len(seq) < window:
        return {'_available': True, 'immunogenic_peptides': [],
                'mean_immunogenicity': 0.0, 'response_freq_pct': 0.0,
                'n_immunogenic': 0}

    peptides = [seq[i:i+window] for i in range(len(seq) - window + 1)]
    result   = ml_tcell_predict(
        peptides, species=species,
        innate_prediction=innate_prediction,
        model_base_dir=model_base_dir,
    )

    if result and result.get('_available'):
        # Add position information to top peptides
        pep_pos = {seq[i:i+window]: i+1 for i in range(len(seq) - window + 1)}
        result['immunogenic_peptides_with_pos'] = [
            (prob, pep, pep_pos.get(pep, 0))
            for prob, pep in result['immunogenic_peptides']
        ]

    return result


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 7: T Cell Integration Test")
    print("=" * 60)

    TEST_CASES = [
        {
            'name':    'SARS-CoV-2 Spike RBD',
            'seq':     'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCY',
            'species': 'human',
        },
        {
            'name':    'HIV gp120 fragment',
            'seq':     'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEAT',
            'species': 'human',
        },
        {
            'name':    'OVA (mouse immunology workhorse)',
            'seq':     'ISQAVHAAHAEINEAGR',
            'species': 'mouse',
        },
    ]

    # Simulate an LNP innate prediction (as Epitrix would provide)
    mock_innate = {
        'TLR7_8':             0.78,
        'DC_maturation':      0.72,
        'antigen_expression': 0.85,
    }

    for tc in TEST_CASES:
        print(f"\nSequence: {tc['name']} [{tc['species']}]")
        result = scan_protein_sequence(
            tc['seq'],
            species=tc['species'],
            window=9,
            innate_prediction=mock_innate,
        )

        if not result or not result.get('_available'):
            err = result.get('error', 'unknown') if result else 'models not found'
            print(f"  ⚠️  Unavailable: {err}")
            print("  Train models first: python 06_train_tcell_model.py")
            continue

        print(f"  Method:             {result['method']}")
        print(f"  Immunogenic/total:  {result['n_immunogenic']} / "
              f"{result.get('n_peptides_scored', '?')}")
        print(f"  Mean immunogenicity:{result['mean_immunogenicity']:.3f}")
        print(f"  Response freq:      {result['response_freq_pct']:.1f}%")
        print(f"  LNP factor:         {result['lnp_factor']:.3f} "
              f"({'applied' if result['lnp_modulated'] else 'default'})")
        print(f"  Top peptides:")
        for prob, pep, pos in result.get('immunogenic_peptides_with_pos', [])[:5]:
            print(f"    pos {pos:>4}  {pep}  prob={prob:.3f}")

    print(f"\n{'='*60}")
    print("✅ Integration test complete.")
    print("\nTo use in app.py:")
    print("  from epitrix_ml.tcell_integrate import scan_protein_sequence")
    print("  result = scan_protein_sequence(sequence, species=species,")
    print("                                 innate_prediction=innate_result)")
    print("=" * 60)
