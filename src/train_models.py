"""
train_models.py — Train all 6 classifiers, cross-validate, tune, evaluate.
Run: .venv\Scripts\python src\train_models.py
Results saved to outputs/results/ and outputs/models/
"""

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              matthews_corrcoef, RocCurveDisplay)

from src.config import (FEATURES_FUSED_CSV, FIGURES_DIR, MODELS_DIR,
                        RESULTS_DIR, RANDOM_SEED)
from src.evaluation import compute_all_metrics

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  BreakHis ML Training Pipeline")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("\n[1/6] Loading fused features...")
df = pd.read_csv(FEATURES_FUSED_CSV)
feat_cols = [c for c in df.columns if c.startswith('glcm_') or c.startswith('lbp_')]

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

X_train = train_df[feat_cols].values.astype(np.float32)
y_train = train_df['label'].values
groups  = train_df['patient_id'].values

X_val   = val_df[feat_cols].values.astype(np.float32)
y_val   = val_df['label'].values

X_test  = test_df[feat_cols].values.astype(np.float32)
y_test  = test_df['label'].values

print(f"   Features    : {len(feat_cols)}")
print(f"   Train       : {X_train.shape[0]} images")
print(f"   Validation  : {X_val.shape[0]} images")
print(f"   Test        : {X_test.shape[0]} images")

# ── Define Classifiers ────────────────────────────────────────────────────────
print("\n[2/6] Setting up classifiers...")
classifiers = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ]),
    'k-NN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'LDA': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=RANDOM_SEED))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1))
    ]),
}
print(f"   Classifiers : {list(classifiers.keys())}")

# ── Cross-Validation ─────────────────────────────────────────────────────────
print("\n[3/6] Running 5-Fold Patient-Aware Cross-Validation...")
cv = StratifiedGroupKFold(n_splits=5)
cv_results = {}

for name, pipeline in classifiers.items():
    scores = cross_val_score(
        pipeline, X_train, y_train,
        groups=groups, cv=cv, scoring='f1', n_jobs=-1
    )
    cv_results[name] = scores
    print(f"   {name:22s}: F1 = {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
print("\n[4/6] Hyperparameter Tuning...")

# SVM
print("   Tuning SVM...")
svm_grid = GridSearchCV(
    classifiers['SVM'],
    {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 'auto'], 'clf__kernel': ['rbf', 'linear']},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
svm_grid.fit(X_train, y_train, groups=groups)
classifiers['SVM'] = svm_grid.best_estimator_
print(f"   Best SVM: {svm_grid.best_params_} | F1={svm_grid.best_score_*100:.2f}%")

# Random Forest
print("   Tuning Random Forest...")
rf_grid = GridSearchCV(
    classifiers['Random Forest'],
    {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [None, 10, 20],
     'clf__min_samples_split': [2, 5]},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
rf_grid.fit(X_train, y_train, groups=groups)
classifiers['Random Forest'] = rf_grid.best_estimator_
print(f"   Best RF : {rf_grid.best_params_} | F1={rf_grid.best_score_*100:.2f}%")

# k-NN
print("   Tuning k-NN...")
knn_grid = GridSearchCV(
    classifiers['k-NN'],
    {'clf__n_neighbors': [3, 5, 7, 11, 15], 'clf__metric': ['euclidean', 'manhattan']},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
knn_grid.fit(X_train, y_train, groups=groups)
classifiers['k-NN'] = knn_grid.best_estimator_
print(f"   Best KNN: {knn_grid.best_params_} | F1={knn_grid.best_score_*100:.2f}%")

# ── Train & Evaluate on Validation ───────────────────────────────────────────
print("\n[5/6] Training all classifiers & evaluating on Validation set...")
val_results  = []
test_results = []
roc_data     = []

for name, pipeline in classifiers.items():
    pipeline.fit(X_train, y_train)

    # Validation
    y_pred_v = pipeline.predict(X_val)
    y_prob_v = pipeline.predict_proba(X_val)[:,1]
    val_metrics = compute_all_metrics(y_val, y_pred_v, y_prob_v, name)
    val_results.append(val_metrics)

    # Test (final holdout)
    y_pred_t = pipeline.predict(X_test)
    y_prob_t = pipeline.predict_proba(X_test)[:,1]
    test_metrics = compute_all_metrics(y_test, y_pred_t, y_prob_t, name)
    test_results.append(test_metrics)

    roc_data.append({'name': name, 'y_true': y_val, 'y_prob': y_prob_v})

    print(f"   {name:22s} | Val  Acc={val_metrics['accuracy']}%  F1={val_metrics['f1']}%  AUC={val_metrics['roc_auc']}%")
    print(f"   {' '*22} | Test Acc={test_metrics['accuracy']}%  F1={test_metrics['f1']}%  AUC={test_metrics['roc_auc']}%")

# ── Save Results ──────────────────────────────────────────────────────────────
val_df_res  = pd.DataFrame(val_results)
test_df_res = pd.DataFrame(test_results)
val_df_res.to_csv( RESULTS_DIR / 'validation_results.csv', index=False)
test_df_res.to_csv(RESULTS_DIR / 'test_results.csv',       index=False)
print(f"\n   Results saved to outputs/results/")

# ── Save Models ───────────────────────────────────────────────────────────────
for name, pipeline in classifiers.items():
    fname = name.lower().replace(' ','_').replace('-','') + '.pkl'
    joblib.dump(pipeline, MODELS_DIR / fname)
print(f"   Models saved to outputs/models/")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating plots...")

# CV bar chart
fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
names  = list(cv_results.keys())
means  = [cv_results[n].mean()*100 for n in names]
stds   = [cv_results[n].std()*100  for n in names]
colors = ['#7c3aed','#2563eb','#059669','#d97706','#dc2626','#db2777']
bars   = ax.bar(names, means, yerr=stds, color=colors, capsize=5,
                edgecolor='#3d3f4e', linewidth=1.5)
for bar, m in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.5,
            f'{m:.1f}%', ha='center', fontsize=10, fontweight='bold', color='white')
ax.set_title('5-Fold CV F1 Scores (Patient-Aware)', fontsize=13, color='white')
ax.set_ylabel('F1 Score (%)', color='white')
ax.tick_params(colors='white'); ax.set_ylim(0, 110)
plt.xticks(rotation=15, color='white')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '06_cv_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
colors2 = ['#7c3aed','#2563eb','#059669','#d97706','#dc2626','#db2777']
fig, ax = plt.subplots(figsize=(9, 7), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
for i, res in enumerate(roc_data):
    RocCurveDisplay.from_predictions(
        res['y_true'], res['y_prob'],
        name=res['name'], ax=ax, color=colors2[i]
    )
ax.plot([0,1],[0,1],'w--', alpha=0.4, label='Random')
ax.set_title('ROC Curves — All Classifiers', fontsize=13, color='white')
ax.legend(loc='lower right', fontsize=9)
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='#0f1117')
fig.suptitle('Confusion Matrices — Validation Set', fontsize=14, color='white')
axes = axes.flatten()
for i, (name, pipeline) in enumerate(classifiers.items()):
    axes[i].set_facecolor('#1a1d27')
    cm = confusion_matrix(y_val, pipeline.predict(X_val))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Malignant'],
                yticklabels=['Benign','Malignant'],
                ax=axes[i], cbar=False,
                annot_kws={'size':14,'weight':'bold'})
    axes[i].set_title(name, fontsize=11, color='white')
    axes[i].tick_params(colors='white')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '06_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print("   Plots saved to outputs/figures/")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
best_val = val_df_res.loc[val_df_res['f1'].idxmax()]
best_test = test_df_res.loc[test_df_res['f1'].idxmax()]
print(f"\n  VALIDATION - Best: {best_val['model']}")
print(f"    Accuracy : {best_val['accuracy']}%")
print(f"    F1-Score : {best_val['f1']}%")
print(f"    ROC-AUC  : {best_val['roc_auc']}%")
print(f"\n  TEST - Best: {best_test['model']}")
print(f"    Accuracy : {best_test['accuracy']}%")
print(f"    F1-Score : {best_test['f1']}%")
print(f"    ROC-AUC  : {best_test['roc_auc']}%")
print("\n  All results: outputs/results/")
print("  All models : outputs/models/")
print("=" * 60)
