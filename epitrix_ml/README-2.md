# Epitrix ML Pipeline

Trains an XGBoost model on IEDB MHC-I binding data to replace
the PSSM epitope scanner in Epitrix with a data-driven predictor.

## Quick Start

```bash
cd epitrix_ml
pip install -r requirements_ml.txt

# Step 1 — Download IEDB data (~500 MB, one-time)
python 01_download_iedb.py

# Step 2 — Process and featurise (~5 min)
python 02_process_mhci.py

# Step 3 — Train XGBoost model (~5-15 min on a laptop)
python 03_train_mhci_model.py

# Step 4 — Test integration
python 04_integrate_epitrix.py
```

## Output files

```
data/
  raw/           IEDB bulk downloads
  processed/     mhci_dataset.parquet

models/
  mhci_xgboost.pkl    Trained model + metadata

results/
  evaluation_report.txt
  confusion_matrix.png
  roc_curves.png
  feature_importance.png
  calibration_curve.png
```

## Plugging into Epitrix

In `app.py`, find `_local_epitope_scan(seq)` and replace with:

```python
from epitrix_ml.integrate import ml_epitope_scan
result = ml_epitope_scan(seq, model_path='epitrix_ml/models/mhci_xgboost.pkl')
```

The function returns the same dict structure — no other changes needed.

## What the model learns

- Input: 9-mer peptide sequence encoded as per-position physicochemical
  features (hydrophobicity, MW, isoelectric point) + one-hot amino acid
  identity + HLA-A*02:01 PSSM score
- Output: 3-class probability (non-binder / weak binder / strong binder)
- Training data: IEDB MHC ligand elution + binding affinity assays
- Benchmark: NetMHCpan 4.1 (via IEDB API)

## Expected performance

On held-out IEDB test data:
- AUC-ROC (binary binder/non-binder): typically 0.88–0.94
- F1 weighted (3-class): typically 0.82–0.89
- Comparable to published PSSM-based tools; below NetMHCpan 4.1

This is a first-generation model. Performance improves with:
1. More training data (multi-allele)
2. Sequence embeddings (ESM-2) instead of hand-crafted features
3. Allele-specific fine-tuning
