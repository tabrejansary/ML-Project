"""
retrain_final.py — Final retraining with:
  - Train+Val combined (more data)
  - SelectKBest (top 30 features, reduces overfitting)
  - class_weight='balanced' (handles class imbalance)
  - Lighter regularization (better generalization)
Run: .venv\Scripts\python src\retrain_final.py
"""

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, roc_curve, auc
)

from src.config import (
    FEATURES_FUSED_CSV, FIGURES_DIR, MODELS_DIR,
    RESULTS_DIR, RANDOM_SEED, CLASS_NAMES
)

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  Final Retraining (Train+Val -> Test)")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_FUSED_CSV)
feat_cols = [c for c in df.columns if c.startswith('glcm_') or c.startswith('lbp_')]

trainval_df = df[df['split'].isin(['train', 'val'])]
test_df     = df[df['split'] == 'test']

X_tv = trainval_df[feat_cols].values.astype(np.float32)
y_tv = trainval_df['label'].values
g_tv = trainval_df['patient_id'].values

X_te = test_df[feat_cols].values.astype(np.float32)
y_te = test_df['label'].values

print(f"\nTrain+Val : {len(X_tv)} images | Benign={(y_tv==0).sum()} Malignant={(y_tv==1).sum()}")
print(f"Test      : {len(X_te)} images | Benign={(y_te==0).sum()} Malignant={(y_te==1).sum()}")

# ── Define Classifiers ────────────────────────────────────────────────────────
K_BEST = 30  # top 30 most discriminative features

classifiers = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', LogisticRegression(
            max_iter=2000, class_weight='balanced',
            C=0.1, random_state=RANDOM_SEED))
    ]),
    'k-NN': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', KNeighborsClassifier(n_neighbors=11, metric='manhattan'))
    ]),
    'LDA': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', LinearDiscriminantAnalysis())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', SVC(
            kernel='rbf', class_weight='balanced',
            C=1, gamma='scale', probability=True,
            random_state=RANDOM_SEED))
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', DecisionTreeClassifier(
            class_weight='balanced', max_depth=10,
            random_state=RANDOM_SEED))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=K_BEST)),
        ('clf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            max_depth=15, random_state=RANDOM_SEED, n_jobs=-1))
    ]),
}

# ── 5-Fold CV on Train+Val ────────────────────────────────────────────────────
print("\n[1] 5-Fold Patient-Aware CV on Train+Val...")
cv = StratifiedGroupKFold(n_splits=5)
cv_results = {}

for name, pipe in classifiers.items():
    scores = cross_val_score(
        pipe, X_tv, y_tv,
        groups=g_tv, cv=cv, scoring='f1', n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name:22s}: F1={scores.mean()*100:.2f}% (+/-{scores.std()*100:.2f}%)")

# ── Train on Full Train+Val, Evaluate on Test ─────────────────────────────────
print("\n[2] Training on Train+Val, evaluating on Test...")

def compute_metrics(name, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    return {
        'Model'      : name,
        'Accuracy'   : round(accuracy_score(y_true, y_pred) * 100, 2),
        'Precision'  : round(precision_score(y_true, y_pred) * 100, 2),
        'Recall'     : round(recall_score(y_true, y_pred) * 100, 2),
        'Specificity': round(spec, 2),
        'F1'         : round(f1_score(y_true, y_pred) * 100, 2),
        'ROC_AUC'    : round(roc_auc_score(y_true, y_prob) * 100, 2),
        'MCC'        : round(matthews_corrcoef(y_true, y_pred), 4),
        'CV_F1'      : round(cv_results[name].mean() * 100, 2),
        'CV_F1_std'  : round(cv_results[name].std() * 100, 2),
    }

test_results  = []
probabilities = {}

for name, pipe in classifiers.items():
    pipe.fit(X_tv, y_tv)

    # Save model
    fname = name.lower().replace(' ', '_').replace('-', '') + '.pkl'
    joblib.dump(pipe, MODELS_DIR / fname)

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    probabilities[name] = y_prob

    m = compute_metrics(name, y_te, y_pred, y_prob)
    test_results.append(m)

    print(f"  {name:22s}: Acc={m['Accuracy']}%  F1={m['F1']}%"
          f"  AUC={m['ROC_AUC']}%  Spec={m['Specificity']}%"
          f"  (CV F1={m['CV_F1']}%)")

results_df = pd.DataFrame(test_results)
results_df.to_csv(RESULTS_DIR / 'test_results.csv', index=False)
results_df.to_csv(RESULTS_DIR / 'validation_results.csv', index=False)

# ── Generate Plots ─────────────────────────────────────────────────────────────
print("\n[3] Generating plots...")
colors = ['#7c3aed', '#2563eb', '#059669', '#d97706', '#dc2626', '#db2777']

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
for i, (name, y_prob) in enumerate(probabilities.items()):
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Test Set', fontsize=13)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / '07_roc_curves_test.png', dpi=150, bbox_inches='tight')
plt.close()

# Best model confusion matrix
best_name = results_df.loc[results_df['F1'].idxmax(), 'Model']
best_pipe = classifiers[best_name]
y_pred_best = best_pipe.predict(X_te)
cm = confusion_matrix(y_te, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Confusion Matrix — {best_name} (Test Set)', fontsize=13)
for ax_i, (data, fmt, title) in enumerate(zip(
    [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100],
    ['d', '.1f'],
    ['Counts', 'Percentages (%)']
)):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[ax_i], cbar=False, annot_kws={'size': 16, 'weight': 'bold'})
    axes[ax_i].set_title(title)
    axes[ax_i].set_xlabel('Predicted')
    axes[ax_i].set_ylabel('Actual')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '07_confusion_matrix_test.png', dpi=150, bbox_inches='tight')
plt.close()

# Per-magnification
mag_results = []
for mag in ['40X', '100X', '200X', '400X']:
    mag_sub = test_df[test_df['magnification'] == mag]
    if len(mag_sub) == 0:
        continue
    X_m = mag_sub[feat_cols].values.astype(np.float32)
    y_m = mag_sub['label'].values
    y_pred_m = best_pipe.predict(X_m)
    y_prob_m = best_pipe.predict_proba(X_m)[:, 1]
    mag_results.append({
        'Magnification': mag,
        'Images': len(mag_sub),
        'Accuracy': round(accuracy_score(y_m, y_pred_m) * 100, 2),
        'F1': round(f1_score(y_m, y_pred_m) * 100, 2),
        'ROC_AUC': round(roc_auc_score(y_m, y_prob_m) * 100, 2),
    })

mag_df = pd.DataFrame(mag_results)
mag_df.to_csv(RESULTS_DIR / 'per_magnification_results.csv', index=False)

print("  Plots saved.")

# ── Final Summary ──────────────────────────────────────────────────────────────
best = results_df.loc[results_df['CV_F1'].idxmax()]
print("\n" + "=" * 60)
print("  RETRAINING COMPLETE!")
print("=" * 60)
print(f"\n  Best model       : {best['Model']}")
print(f"  CV F1 (reported) : {best['CV_F1']}% (+/-{best['CV_F1_std']}%)")
print(f"  Test Accuracy    : {best['Accuracy']}%")
print(f"  Test Precision   : {best['Precision']}%")
print(f"  Test Recall      : {best['Recall']}%")
print(f"  Test Specificity : {best['Specificity']}%")
print(f"  Test F1          : {best['F1']}%")
print(f"  Test ROC-AUC     : {best['ROC_AUC']}%")
print(f"  Test MCC         : {best['MCC']}")
print()
print(f"  Per-magnification results saved.")
print(f"  All 6 models saved to outputs/models/")
print("=" * 60)
