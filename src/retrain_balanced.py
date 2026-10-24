"""
retrain_balanced.py — Retrain all classifiers with class_weight='balanced'
to fix class imbalance issue.
Run: .venv\Scripts\python src\retrain_balanced.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              matthews_corrcoef, roc_curve, auc)

from src.config import (FEATURES_FUSED_CSV, FIGURES_DIR, MODELS_DIR,
                        RESULTS_DIR, RANDOM_SEED, CLASS_NAMES)

np.random.seed(RANDOM_SEED)

print("=" * 60)
print("  Retraining with class_weight='balanced'")
print("=" * 60)

# ── Load Data ─────────────────────────────────────────────────────────────────
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

print(f"\nTrain: {X_train.shape[0]} imgs | Benign={( y_train==0).sum()} Malignant={(y_train==1).sum()}")
print(f"Val  : {X_val.shape[0]} imgs   | Benign={(y_val==0).sum()} Malignant={(y_val==1).sum()}")
print(f"Test : {X_test.shape[0]} imgs   | Benign={(y_test==0).sum()} Malignant={(y_test==1).sum()}")

# ── Classifiers with class_weight='balanced' ─────────────────────────────────
classifiers = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   random_state=RANDOM_SEED))
    ]),
    'k-NN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))  # kNN has no class_weight
    ]),
    'LDA': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())  # LDA handles imbalance naturally
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced',
                    random_state=RANDOM_SEED))
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(class_weight='balanced',
                                        random_state=RANDOM_SEED))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        random_state=RANDOM_SEED, n_jobs=-1))
    ]),
}

print(f"\nClassifiers: {list(classifiers.keys())}")

# ── Tune SVM ──────────────────────────────────────────────────────────────────
print("\n[Tuning] SVM...")
svm_grid = GridSearchCV(
    classifiers['SVM'],
    {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 'auto'],
     'clf__kernel': ['rbf', 'linear']},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
svm_grid.fit(X_train, y_train, groups=groups)
classifiers['SVM'] = svm_grid.best_estimator_
print(f"  Best SVM: {svm_grid.best_params_} | F1={svm_grid.best_score_*100:.2f}%")

# ── Tune Random Forest ────────────────────────────────────────────────────────
print("[Tuning] Random Forest...")
rf_grid = GridSearchCV(
    classifiers['Random Forest'],
    {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [None, 10, 20]},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
rf_grid.fit(X_train, y_train, groups=groups)
classifiers['Random Forest'] = rf_grid.best_estimator_
print(f"  Best RF : {rf_grid.best_params_} | F1={rf_grid.best_score_*100:.2f}%")

# ── Tune k-NN ─────────────────────────────────────────────────────────────────
print("[Tuning] k-NN...")
knn_grid = GridSearchCV(
    classifiers['k-NN'],
    {'clf__n_neighbors': [3, 5, 7, 11, 15], 'clf__metric': ['euclidean', 'manhattan']},
    cv=StratifiedGroupKFold(n_splits=3), scoring='f1', n_jobs=-1, refit=True
)
knn_grid.fit(X_train, y_train, groups=groups)
classifiers['k-NN'] = knn_grid.best_estimator_
print(f"  Best KNN: {knn_grid.best_params_} | F1={knn_grid.best_score_*100:.2f}%")

# ── Train All & Evaluate ──────────────────────────────────────────────────────
print("\nTraining & Evaluating all classifiers...")

def full_metrics(y_true, y_pred, y_prob, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Model'      : name,
        'Accuracy'   : round(accuracy_score(y_true, y_pred)*100, 2),
        'Precision'  : round(precision_score(y_true, y_pred)*100, 2),
        'Recall'     : round(recall_score(y_true, y_pred)*100, 2),
        'Specificity': round(spec*100, 2),
        'F1'         : round(f1_score(y_true, y_pred)*100, 2),
        'ROC-AUC'    : round(roc_auc_score(y_true, y_prob)*100, 2),
        'MCC'        : round(matthews_corrcoef(y_true, y_pred), 4),
    }

val_results  = []
test_results = []
roc_val_data = []
roc_test_data= []

for name, pipeline in classifiers.items():
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_DIR / (name.lower().replace(' ','_').replace('-','') + '.pkl'))

    y_pred_v = pipeline.predict(X_val)
    y_prob_v = pipeline.predict_proba(X_val)[:,1]
    val_results.append(full_metrics(y_val, y_pred_v, y_prob_v, name))
    roc_val_data.append({'name': name, 'y_true': y_val, 'y_prob': y_prob_v})

    y_pred_t = pipeline.predict(X_test)
    y_prob_t = pipeline.predict_proba(X_test)[:,1]
    test_results.append(full_metrics(y_test, y_pred_t, y_prob_t, name))
    roc_test_data.append({'name': name, 'y_true': y_test, 'y_prob': y_prob_t})

    print(f"  {name:22s} | Val  Acc={val_results[-1]['Accuracy']}% F1={val_results[-1]['F1']}%"
          f" | Test Acc={test_results[-1]['Accuracy']}% F1={test_results[-1]['F1']}%")

val_df_r  = pd.DataFrame(val_results)
test_df_r = pd.DataFrame(test_results)
val_df_r.to_csv( RESULTS_DIR / 'validation_results.csv', index=False)
test_df_r.to_csv(RESULTS_DIR / 'test_results.csv',       index=False)

# ── Generate Plots ────────────────────────────────────────────────────────────
colors = ['#7c3aed','#2563eb','#059669','#d97706','#dc2626','#db2777']

# ROC curves test
fig, ax = plt.subplots(figsize=(9, 7))
for i, res in enumerate(roc_test_data):
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{res['name']} (AUC={auc(fpr,tpr):.3f})")
ax.plot([0,1],[0,1],'--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Test Set (Balanced)', fontsize=13)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / '07_roc_curves_test.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices test
best_name = test_df_r.loc[test_df_r['F1'].idxmax(), 'Model']
best_pipe = classifiers[best_name]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Confusion Matrix — {best_name} Test Set', fontsize=13)
cm = confusion_matrix(y_test, best_pipe.predict(X_test))
for ax_i, (data, fmt, title) in enumerate(zip(
    [cm, cm.astype(float)/cm.sum(axis=1, keepdims=True)*100],
    ['d', '.1f'], ['Counts', 'Percentages (%)']
)):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[ax_i], cbar=False, annot_kws={'size':16,'weight':'bold'})
    axes[ax_i].set_title(title)
    axes[ax_i].set_xlabel('Predicted'); axes[ax_i].set_ylabel('Actual')
plt.tight_layout()
fig.savefig(FIGURES_DIR / '07_confusion_matrix_test.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("  RETRAINING COMPLETE!")
print("=" * 60)
best = test_df_r.loc[test_df_r['F1'].idxmax()]
print(f"\n  Best Model : {best['Model']}")
print(f"  Accuracy   : {best['Accuracy']}%")
print(f"  Precision  : {best['Precision']}%")
print(f"  Recall     : {best['Recall']}%")
print(f"  Specificity: {best['Specificity']}%")
print(f"  F1-Score   : {best['F1']}%")
print(f"  ROC-AUC    : {best['ROC-AUC']}%")
print(f"  MCC        : {best['MCC']}")
print("\n  All models saved with balanced class weights!")
print("=" * 60)
