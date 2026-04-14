"""
EPITRIX ML PIPELINE — STEP 3
=============================
Trains an XGBoost model on the processed MHC-I dataset.

What this does:
  1. Loads the processed dataset from data/processed/mhci_dataset.parquet
  2. Splits into train / validation / test sets (70/15/15 stratified)
  3. Trains an XGBoost classifier (3-class: non-binder / weak / strong)
  4. Evaluates on the held-out test set with full metrics
  5. Saves the trained model to models/mhci_xgboost.pkl
  6. Generates evaluation plots to results/

Usage:
    python 03_train_mhci_model.py

Expected input:  data/processed/mhci_dataset.parquet  (from step 2)
Expected output: models/mhci_xgboost.pkl
                 results/evaluation_report.txt
                 results/confusion_matrix.png
                 results/feature_importance.png
                 results/calibration_curve.png
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, matthews_corrcoef, f1_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models");    MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR   = Path("results");   RESULTS_DIR.mkdir(exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150,
})
COLORS = ['#6b7280', '#f59e0b', '#2563eb']
CLASS_NAMES = ['Non-binder', 'Weak binder', 'Strong binder']


def load_data(dataset_name: str = 'human') -> tuple:
    """Load processed dataset. dataset_name: 'human', 'mouse', or 'combined'."""
    # Try new species-specific filename first, fall back to legacy name
    path = PROCESSED_DIR / f"mhci_dataset_{dataset_name}.parquet"
    if not path.exists() and dataset_name == 'human':
        path = PROCESSED_DIR / "mhci_dataset.parquet"   # legacy fallback

    if not path.exists():
        return None, None, None

    print(f"  Loading: {path}")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")

    META_COLS = {'label_binary', 'label_3class', 'peptide', 'allele', 'species'}
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].values.astype(np.float32)
    y = df['label_3class'].values.astype(int)

    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")

    return X, y, feature_cols


def train_one_model(dataset_name: str, label: str):
    """Train, evaluate, and save one species model."""
    print(f"\n{'='*60}")
    print(f"  Training: {label} model  [{dataset_name}]")
    print(f"{'='*60}")

    results_sub = RESULTS_DIR / dataset_name
    results_sub.mkdir(exist_ok=True)

    # Load
    X, y, feature_cols = load_data(dataset_name)
    if X is None:
        print(f"  ⚠️  No dataset found for '{dataset_name}' — skipping.")
        return

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Train
    print(f"  Training XGBoost...")
    model = train_model(X_train, y_train)

    # Evaluate
    metrics, y_pred, y_proba, y_binary_true, y_binary_prob = evaluate_model(
        model, X_test, y_test, feature_cols
    )

    # Save model
    model_data = {
        'model':        model,
        'feature_cols': feature_cols,
        'class_names':  CLASS_NAMES,
        'metrics':      {k: v for k, v in metrics.items() if k != 'class_report'},
        'trained_at':   datetime.now().isoformat(),
        'n_train':      len(X_train),
        'species':      dataset_name,
        'xgb_version':  xgb.__version__,
    }
    model_path = MODELS_DIR / f"mhci_xgboost_{dataset_name}.pkl"
    joblib.dump(model_data, model_path, compress=3)
    print(f"  Saved: {model_path}  ({model_path.stat().st_size/1e6:.1f} MB)")

    # Plots
    plot_confusion_matrix(np.array(metrics['confusion_matrix']),
                          results_sub / "confusion_matrix.png")
    plot_roc_curves(y_test, y_proba, results_sub / "roc_curves.png")
    plot_feature_importance(model, feature_cols, results_sub / "feature_importance.png")
    plot_calibration(y_binary_true, y_binary_prob, results_sub / "calibration_curve.png")
    save_evaluation_report(metrics, results_sub / "evaluation_report.txt")

    metrics_out = {k: v for k, v in metrics.items() if k not in ('class_report',)}
    with open(results_sub / "metrics.json", 'w') as f:
        json.dump(metrics_out, f, indent=2)

    return metrics


def main():
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 3: Train All Species Models")
    print("  Models: human (HLA), mouse (H-2), combined")
    print("=" * 60)

    MODELS = [
        ('human',    'Human HLA'),
        ('mouse',    'Mouse H-2'),
        ('combined', 'Combined (human + mouse)'),
    ]

    summary = {}
    for dataset_name, label in MODELS:
        result = train_one_model(dataset_name, label)
        if result:
            summary[label] = {
                'auc_roc': result['binary_auc_roc'],
                'f1_w':    result['f1_weighted'],
                'mcc':     result['mcc'],
            }

    print(f"\n{'='*60}")
    print("✅ ALL MODELS TRAINED")
    print(f"{'='*60}")
    print(f"\n{'Model':<30} {'AUC-ROC':>8} {'F1(w)':>8} {'MCC':>8}")
    print(f"{'─'*58}")
    for label, m in summary.items():
        print(f"  {label:<28} {m['auc_roc']:>8.4f} {m['f1_w']:>8.4f} {m['mcc']:>8.4f}")

    print(f"\n  Models saved in: {MODELS_DIR}/")
    print(f"    mhci_xgboost_human.pkl")
    print(f"    mhci_xgboost_mouse.pkl")
    print(f"    mhci_xgboost_combined.pkl")
    print(f"\n  Plots saved in: {RESULTS_DIR}/human/  mouse/  combined/")
    print(f"\n  Next: Run 04_integrate_epitrix.py")
    print("=" * 60)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost with class weighting and early stopping.
    Hyperparameters chosen for good performance on tabular immunology data
    without requiring a GPU.
    """
    # Compute class weights to handle imbalance
    class_counts = np.bincount(y_train)
    total        = len(y_train)
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])

    model = xgb.XGBClassifier(
        # Architecture
        n_estimators        = 500,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,
        gamma               = 0.1,
        reg_alpha           = 0.1,   # L1
        reg_lambda          = 1.0,   # L2
        # Multi-class
        objective           = 'multi:softprob',
        num_class           = 3,
        eval_metric         = ['mlogloss', 'merror'],
        # Reproducibility
        random_state        = 42,
        # Performance
        n_jobs              = -1,
        tree_method         = 'hist',
        # Early stopping
        early_stopping_rounds = 30,
        verbosity           = 0,
    )

    # Split off a small validation set for early stopping
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.1, random_state=42, stratify=y_train
    )

    model.fit(
        X_tr, y_tr,
        sample_weight    = w_tr,
        eval_set         = [(X_val, y_val)],
        verbose          = False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, feature_cols: list) -> dict:
    """Comprehensive evaluation with all relevant metrics."""

    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)

    # ── Core metrics ──────────────────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)
    mcc    = matthews_corrcoef(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro')
    f1_wei = f1_score(y_test, y_pred, average='weighted')

    # ── AUC-ROC (one-vs-rest) ─────────────────────────────────────────────────
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_roc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')

    # ── Binary metrics (binder vs non-binder — clinically most relevant) ──────
    y_binary_true = (y_test > 0).astype(int)
    y_binary_prob = y_proba[:, 1] + y_proba[:, 2]   # weak + strong binder probability
    y_binary_pred = (y_binary_prob > 0.5).astype(int)
    auc_binary    = roc_auc_score(y_binary_true, y_binary_prob)
    ap_binary     = average_precision_score(y_binary_true, y_binary_prob)

    metrics = {
        'n_test':              len(y_test),
        'accuracy':            report['accuracy'],
        'f1_macro':            f1_mac,
        'f1_weighted':         f1_wei,
        'mcc':                 mcc,
        'auc_roc_macro_ovr':   auc_roc,
        'binary_auc_roc':      auc_binary,
        'binary_avg_precision':ap_binary,
        'class_report':        report,
        'confusion_matrix':    cm.tolist(),
    }

    return metrics, y_pred, y_proba, y_binary_true, y_binary_prob


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('MHC-I Binding Prediction — Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curves(y_test, y_proba, out_path: Path):
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — MHC-I Binding Prediction', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_feature_importance(model, feature_cols: list, out_path: Path, top_n: int = 30):
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    top = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(range(len(top)), top.values[::-1], color='#2563eb', alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features\nMHC-I XGBoost Model', fontsize=13, fontweight='bold', pad=15)

    # Annotate bars
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', fontsize=8, color='#374151')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration(y_binary_true, y_binary_prob, out_path: Path):
    prob_true, prob_pred = calibration_curve(y_binary_true, y_binary_prob, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    axes[0].plot(prob_pred, prob_true, 's-', color='#2563eb', lw=2, label='XGBoost')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect calibration')
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Fraction of Positives', fontsize=12)
    axes[0].set_title('Calibration Curve\n(Binder vs Non-binder)', fontsize=12, fontweight='bold')
    axes[0].legend(framealpha=0.9)

    # Score distribution
    axes[1].hist(y_binary_prob[y_binary_true == 0], bins=40, alpha=0.6,
                 color='#6b7280', label='Non-binder', density=True)
    axes[1].hist(y_binary_prob[y_binary_true == 1], bins=40, alpha=0.6,
                 color='#2563eb', label='Binder', density=True)
    axes[1].axvline(0.5, color='#ef4444', lw=1.5, linestyle='--', label='Threshold = 0.5')
    axes[1].set_xlabel('Predicted Binding Probability', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Score Distribution\nby True Label', fontsize=12, fontweight='bold')
    axes[1].legend(framealpha=0.9)

    plt.suptitle('MHC-I Binding Prediction — Model Calibration', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def save_evaluation_report(metrics: dict, out_path: Path):
    """Write a human-readable evaluation report."""
    lines = [
        "=" * 60,
        "EPITRIX MHC-I MODEL — EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "── Test Set Summary ────────────────────────────────────",
        f"  Samples:              {metrics['n_test']:,}",
        f"  Accuracy:             {metrics['accuracy']:.4f}",
        f"  F1 (macro):           {metrics['f1_macro']:.4f}",
        f"  F1 (weighted):        {metrics['f1_weighted']:.4f}",
        f"  MCC:                  {metrics['mcc']:.4f}",
        f"  AUC-ROC (macro OvR):  {metrics['auc_roc_macro_ovr']:.4f}",
        "",
        "── Binary Metrics (Binder vs Non-binder) ───────────────",
        f"  AUC-ROC:              {metrics['binary_auc_roc']:.4f}",
        f"  Average Precision:    {metrics['binary_avg_precision']:.4f}",
        "",
        "── Per-class Metrics ───────────────────────────────────",
    ]

    report = metrics['class_report']
    for cls_name in CLASS_NAMES:
        cls_key = cls_name
        if cls_key in report:
            r = report[cls_key]
            lines.append(f"  {cls_name:<18} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={r['support']:,}")

    lines += [
        "",
        "── Confusion Matrix ────────────────────────────────────",
        "  Rows=True, Cols=Predicted",
        "  Classes: 0=Non-binder, 1=Weak binder, 2=Strong binder",
    ]
    cm = np.array(metrics['confusion_matrix'])
    for row in cm:
        lines.append("  " + "  ".join(f"{v:>7,}" for v in row))

    lines += [
        "",
        "── Interpretation ──────────────────────────────────────",
        "  AUC-ROC >0.90: Excellent discrimination",
        "  AUC-ROC >0.80: Good — suitable for research use",
        "  AUC-ROC >0.70: Moderate — use with caution",
        "  MCC >0.60:     Strong correlation with true labels",
        "",
        "=" * 60,
    ]

    text = "\n".join(lines)
    with open(out_path, 'w') as f:
        f.write(text)

    print(f"\n{text}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
