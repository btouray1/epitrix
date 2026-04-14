"""
EPITRIX ML PIPELINE — STEP 6
==============================
Trains XGBoost models for T cell immunogenicity prediction.

Two models are trained:
  1. Binary classifier   — Positive vs Negative T cell response
  2. Regression model    — Response frequency % (0-100)

The regression model is more informative when available.
Both are saved and Epitrix uses the regression output where possible,
falling back to binary classification.

Usage:
    python 06_train_tcell_model.py

Expected input:  data/processed/tcell_dataset.parquet  (from step 5)
Expected output: models/tcell_xgboost_classifier.pkl
                 models/tcell_xgboost_regressor.pkl
                 results/tcell/evaluation_report.txt
                 results/tcell/*.png
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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, matthews_corrcoef, f1_score,
    mean_absolute_error, r2_score
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models");   MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR   = Path("results/tcell"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150,
})


def load_data() -> tuple:
    path = PROCESSED_DIR / "tcell_dataset.parquet"
    print(f"  Loading: {path}")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")

    META_COLS = {'label_binary', 'response_freq', 'peptide', 'allele', 'n_subjects'}
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X        = df[feature_cols].fillna(0).values.astype(np.float32)
    y_binary = df['label_binary'].values.astype(int)
    y_freq   = df['response_freq'].values.astype(float)
    weights  = np.log1p(df['n_subjects'].values).astype(np.float32)

    # ── Species masks for species-specific classifiers ────────────────────────
    human_mask = df['is_human'].values == 1
    mouse_mask = df['is_mouse'].values == 1

    X_human = X[human_mask];  y_human = y_binary[human_mask]; w_human = weights[human_mask]
    X_mouse = X[mouse_mask];  y_mouse = y_binary[mouse_mask]; w_mouse = weights[mouse_mask]

    # ── Continuous subset for regressor (combined — mouse has too few) ────────
    continuous_mask = (y_freq > 1) & (y_freq < 99)
    X_cont = X[continuous_mask]
    y_cont = y_freq[continuous_mask]
    w_cont = weights[continuous_mask]

    print(f"  Features:             {X.shape[1]}")
    print(f"  Total assays:         {len(df):,}")
    print(f"  Human:                {human_mask.sum():,} "
          f"({human_mask.mean()*100:.1f}%)")
    print(f"  Mouse:                {mouse_mask.sum():,} "
          f"({mouse_mask.mean()*100:.1f}%)")
    print(f"  Positive (all):       {y_binary.sum():,} ({y_binary.mean()*100:.1f}%)")
    print(f"  Continuous freq rows: {continuous_mask.sum():,} — used for regressor")

    return (X, y_binary, y_freq, weights, feature_cols,
            X_human, y_human, w_human,
            X_mouse, y_mouse, w_mouse,
            X_cont,  y_cont,  w_cont)


def train_classifier(X_train, y_train, w_train) -> xgb.XGBClassifier:
    """Binary classifier: Positive vs Negative T cell response."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, w_train,
        test_size=0.1, random_state=42, stratify=y_train
    )

    model = xgb.XGBClassifier(
        n_estimators        = 500,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,
        gamma               = 0.1,
        reg_alpha           = 0.1,
        reg_lambda          = 1.0,
        scale_pos_weight    = pos_weight,
        objective           = 'binary:logistic',
        eval_metric         = ['logloss', 'auc'],
        random_state        = 42,
        n_jobs              = -1,
        tree_method         = 'hist',
        early_stopping_rounds = 30,
        verbosity           = 0,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Classifier best iteration: {model.best_iteration}")
    return model


def train_regressor(X_train, y_train, w_train) -> xgb.XGBRegressor:
    """
    Regression model trained ONLY on rows with continuous response frequency.
    Uses arcsin-sqrt transform (variance-stabilising for proportion data).
    This avoids the bimodal 0/100 distribution from binary-only papers.
    """
    # Arcsin-sqrt transform: variance-stabilising for proportions
    y_prop     = y_train / 100.0
    y_trans    = np.arcsin(np.sqrt(np.clip(y_prop, 0, 1)))

    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_trans, w_train,
        test_size=0.1, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators        = 300,
        max_depth           = 4,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 15,
        gamma               = 0.3,
        reg_alpha           = 0.5,
        reg_lambda          = 3.0,
        objective           = 'reg:squarederror',
        eval_metric         = 'rmse',
        random_state        = 42,
        n_jobs              = -1,
        tree_method         = 'hist',
        early_stopping_rounds = 30,
        verbosity           = 0,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Regressor best iteration:  {model.best_iteration}")
    print(f"  Regressor trained on {len(X_train):,} continuous-response rows")
    return model


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    rep = classification_report(y_test, y_pred,
                                 target_names=['Negative','Positive'],
                                 output_dict=True)

    return {
        'n_test':      len(y_test),
        'auc_roc':     float(auc),
        'avg_prec':    float(ap),
        'mcc':         float(mcc),
        'f1_weighted': float(f1),
        'accuracy':    float(rep['accuracy']),
        'report':      rep,
        'y_pred':      y_pred,
        'y_proba':     y_proba,
    }


def evaluate_regressor(model, X_test, y_test_freq) -> dict:
    """Evaluate regressor. Back-transforms arcsin-sqrt predictions to %."""
    y_trans_pred = model.predict(X_test)
    # Back-transform: arcsin-sqrt -> proportion -> percentage
    y_pred = np.clip(np.sin(y_trans_pred) ** 2 * 100, 0, 100)

    mae  = mean_absolute_error(y_test_freq, y_pred)
    r2   = r2_score(y_test_freq, y_pred)
    rmse = float(np.sqrt(np.mean((y_test_freq - y_pred) ** 2)))
    corr = float(np.corrcoef(y_test_freq, y_pred)[0, 1])

    return {
        'mae':    float(mae),
        'rmse':   float(rmse),
        'r2':     float(r2),
        'corr':   float(corr),
        'y_pred': y_pred,
    }


def plot_classifier_results(clf_metrics, results_dir):
    y_test  = clf_metrics['_y_test']
    y_proba = clf_metrics['y_proba']
    y_pred  = clf_metrics['y_pred']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color='#2563eb', lw=2,
                 label=f"AUC = {clf_metrics['auc_roc']:.3f}")
    axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve — T Cell Immunogenicity')
    axes[0].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'])
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    axes[2].plot(prob_pred, prob_true, 's-', color='#2563eb', lw=2, label='Model')
    axes[2].plot([0,1],[0,1],'k--',lw=1,alpha=0.6,label='Perfect')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction Positive')
    axes[2].set_title('Calibration Curve')
    axes[2].legend()

    plt.suptitle('T Cell Immunogenicity Classifier', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(results_dir / "classifier_evaluation.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/classifier_evaluation.png")


def plot_regressor_results(reg_metrics, y_test_freq, results_dir):
    y_pred = reg_metrics['y_pred']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: predicted vs actual
    axes[0].scatter(y_test_freq, y_pred, alpha=0.3, s=8, color='#2563eb')
    lim = [0, 100]
    axes[0].plot(lim, lim, 'r--', lw=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual Response Frequency (%)')
    axes[0].set_ylabel('Predicted Response Frequency (%)')
    axes[0].set_title(f"Predicted vs Actual\nR²={reg_metrics['r2']:.3f}  "
                      f"MAE={reg_metrics['mae']:.1f}%  "
                      f"r={reg_metrics['corr']:.3f}")
    axes[0].legend()

    # Residuals
    residuals = y_pred - y_test_freq
    axes[1].hist(residuals, bins=50, color='#2563eb', alpha=0.7)
    axes[1].axvline(0, color='red', lw=1.5, linestyle='--')
    axes[1].set_xlabel('Residual (Predicted − Actual)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f"Residual Distribution\nRMSE={reg_metrics['rmse']:.1f}%")

    plt.suptitle('T Cell Response Frequency Regressor', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(results_dir / "regressor_evaluation.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/regressor_evaluation.png")


def plot_feature_importance(model, feature_cols, results_dir, title, fname, top_n=25):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top)), top.values[::-1], color='#2563eb', alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'Top {top_n} Features — {title}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(results_dir / fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_dir}/{fname}")


def save_report(clf_m, reg_m, results_dir):
    lines = [
        "=" * 60,
        "EPITRIX T CELL IMMUNOGENICITY — EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "── Binary Classifier (Positive vs Negative) ────────────",
        f"  Test samples:     {clf_m['n_test']:,}",
        f"  AUC-ROC:          {clf_m['auc_roc']:.4f}",
        f"  Avg Precision:    {clf_m['avg_prec']:.4f}",
        f"  MCC:              {clf_m['mcc']:.4f}",
        f"  F1 (weighted):    {clf_m['f1_weighted']:.4f}",
        f"  Accuracy:         {clf_m['accuracy']:.4f}",
        "",
        "── Response Frequency Regressor (0-100%) ───────────────",
        f"  MAE:              {reg_m['mae']:.2f}%",
        f"  RMSE:             {reg_m['rmse']:.2f}%",
        f"  R²:               {reg_m['r2']:.4f}",
        f"  Pearson r:        {reg_m['corr']:.4f}",
        "",
        "── Per-class Metrics (Classifier) ──────────────────────",
    ]
    rep = clf_m['report']
    for cls in ['Negative', 'Positive']:
        if cls in rep:
            r = rep[cls]
            lines.append(
                f"  {cls:<12} P={r['precision']:.3f}  "
                f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
                f"n={r['support']:,}"
            )
    lines += [
        "",
        "── Interpretation ──────────────────────────────────────",
        "  AUC-ROC >0.80: Good discrimination",
        "  R² >0.50:      Regression explains majority of variance",
        "  MAE <20%:      Mean prediction error under 20 percentage points",
        "",
        "── LNP integration note ────────────────────────────────",
        "  This model predicts T cell immunogenicity from sequence +",
        "  delivery class + allele. In Epitrix, the output is further",
        "  modulated by LNP-specific factors (TLR7/8 activation,",
        "  DC maturation, antigen expression) from the mechanistic",
        "  cascade model.",
        "=" * 60,
    ]
    text = "\n".join(lines)
    with open(results_dir / "evaluation_report.txt", 'w') as f:
        f.write(text)
    print(f"\n{text}")


def train_and_save_classifier(X_tr, y_tr, w_tr, X_te, y_te,
                               feature_cols, name, results_sub):
    """Train, evaluate, save one classifier and its plots."""
    results_sub.mkdir(parents=True, exist_ok=True)

    clf = train_classifier(X_tr, y_tr, w_tr)
    metrics = evaluate_classifier(clf, X_te, y_te)
    metrics['_y_test'] = y_te

    clf_data = {
        'model':        clf,
        'feature_cols': feature_cols,
        'task':         'binary_classification',
        'species':      name,
        'metrics':      {k: v for k, v in metrics.items()
                         if k not in ('report','y_pred','y_proba','_y_test')},
        'trained_at':   datetime.now().isoformat(),
        'n_train':      len(X_tr),
        'xgb_version':  xgb.__version__,
    }
    path = MODELS_DIR / f"tcell_xgboost_classifier_{name}.pkl"
    joblib.dump(clf_data, path, compress=3)
    print(f"  Saved: {path}  ({path.stat().st_size/1e6:.1f} MB)")

    plot_classifier_results(metrics, results_sub)
    plot_feature_importance(clf, feature_cols, results_sub,
                            f"T Cell Classifier ({name})",
                            "feature_importance_classifier.png")
    save_report(metrics, {'mae': 0, 'rmse': 0, 'r2': 0, 'corr': 0},
                results_sub)

    return metrics, clf


def main():
    print("=" * 60)
    print("EPITRIX ML PIPELINE — Step 6: Train T Cell Models")
    print("  Training: human, mouse, combined classifiers + regressor")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    (X, y_binary, y_freq, weights, feature_cols,
     X_human, y_human, w_human,
     X_mouse, y_mouse, w_mouse,
     X_cont,  y_cont,  w_cont) = load_data()

    print("\n[2/4] Splitting data...")

    def split(X_, y_, w_, stratify=True):
        strat = y_ if stratify else None
        Xte, Xtr, yte, ytr, wte, wtr = train_test_split(
            X_, y_, w_, test_size=0.85, random_state=42, stratify=strat)
        return Xtr, Xte, ytr, yte, wtr, wte

    X_h_tr, X_h_te, y_h_tr, y_h_te, w_h_tr, _ = split(X_human, y_human, w_human)
    X_m_tr, X_m_te, y_m_tr, y_m_te, w_m_tr, _ = split(X_mouse, y_mouse, w_mouse)

    X_temp, X_all_te, y_temp, y_all_te, w_temp, _ = train_test_split(
        X, y_binary, weights, test_size=0.15, random_state=42, stratify=y_binary)
    X_all_tr, _, y_all_tr, _, w_all_tr, _ = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.15/0.85,
        random_state=42, stratify=y_temp)

    X_c_tr, X_c_te, y_c_tr, y_c_te, w_c_tr, _ = train_test_split(
        X_cont, y_cont, w_cont, test_size=0.15, random_state=42)

    print(f"  Human classifier:    train {len(X_h_tr):,}  test {len(X_h_te):,}")
    print(f"  Mouse classifier:    train {len(X_m_tr):,}  test {len(X_m_te):,}")
    print(f"  Combined classifier: train {len(X_all_tr):,}  test {len(X_all_te):,}")
    print(f"  Regressor (cont.):   train {len(X_c_tr):,}  test {len(X_c_te):,}")

    print("\n[3/4] Training models...")
    summary = {}

    for label, Xtr, ytr, wtr, Xte, yte in [
        ('human',    X_h_tr,   y_h_tr,   w_h_tr,   X_h_te,   y_h_te),
        ('mouse',    X_m_tr,   y_m_tr,   w_m_tr,   X_m_te,   y_m_te),
        ('combined', X_all_tr, y_all_tr, w_all_tr, X_all_te, y_all_te),
    ]:
        print(f"\n  --- {label.upper()} classifier ---")
        metrics, _ = train_and_save_classifier(
            Xtr, ytr, wtr, Xte, yte,
            feature_cols, label,
            RESULTS_DIR / label
        )
        summary[label] = metrics

    print(f"\n  --- REGRESSOR (combined continuous) ---")
    reg = train_regressor(X_c_tr, y_c_tr, w_c_tr)
    reg_metrics = evaluate_regressor(reg, X_c_te, y_c_te)

    reg_data = {
        'model':        reg,
        'feature_cols': feature_cols,
        'task':         'response_frequency_regression',
        'species':      'combined',
        'metrics':      {k: v for k, v in reg_metrics.items() if k != 'y_pred'},
        'trained_at':   datetime.now().isoformat(),
        'n_train':      len(X_c_tr),
        'xgb_version':  xgb.__version__,
    }
    reg_path = MODELS_DIR / "tcell_xgboost_regressor.pkl"
    joblib.dump(reg_data, reg_path, compress=3)
    print(f"  Saved: {reg_path}  ({reg_path.stat().st_size/1e6:.1f} MB)")

    plot_regressor_results(reg_metrics, y_c_te, RESULTS_DIR / 'combined')
    plot_feature_importance(reg, feature_cols, RESULTS_DIR / 'combined',
                            "T Cell Regressor", "feature_importance_regressor.png")

    print(f"\n[4/4] Summary")
    print(f"\n{'='*60}")
    print("✅ ALL T CELL MODELS TRAINED")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'AUC-ROC':>8} {'F1(w)':>8} {'MCC':>8}")
    print(f"{'─'*50}")
    for label, m in summary.items():
        print(f"  {label:<18} {m['auc_roc']:>8.4f} {m['f1_weighted']:>8.4f} "
              f"{m['mcc']:>8.4f}")
    print(f"\n  Regressor (continuous freq):")
    print(f"    R²={reg_metrics['r2']:.4f}  "
          f"MAE={reg_metrics['mae']:.1f}%  "
          f"Pearson r={reg_metrics['corr']:.4f}")
    print(f"\n  Models in: {MODELS_DIR}/")
    print(f"    tcell_xgboost_classifier_human.pkl")
    print(f"    tcell_xgboost_classifier_mouse.pkl")
    print(f"    tcell_xgboost_classifier_combined.pkl")
    print(f"    tcell_xgboost_regressor.pkl")
    print(f"\n  Next: Run 07_integrate_tcell.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
