# HistoCAD — Breast Histopathology Diagnostic System

A Machine Learning pipeline and diagnostic workstation for breast histopathology image classification using GLCM (Gray-Level Co-occurrence Matrix) and LBP (Local Binary Patterns) texture features with classical ML classifiers on the BreaKHis dataset.

---

## 🔬 Key Features

- **Multi-Feature Fusion**: Concatenation of GLCM statistical texture features and multi-scale Local Binary Patterns (LBP).
- **Classifiers Implemented**:
  - Random Forest (Top Performer)
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (k-NN)
  - Decision Tree
  - Logistic Regression
  - Linear Discriminant Analysis (LDA)
- **Interactive Web Diagnostic Workstation**: Powered by Streamlit with clinical analytics, probability distributions, radar charts, and live prediction on uploaded biopsies.
- **Explainable AI (XAI)**: SHAP-based feature importance interpretation.

---

## 📁 Project Structure

```
├── app.py                     # Streamlit Diagnostic Web Application
├── requirements.txt           # Project dependencies
├── metadata.csv               # Image-level dataset metadata & patient mapping
├── features_fused.csv         # Extracted fused GLCM + LBP features
├── features_glcm.csv          # Extracted GLCM features
├── features_lbp.csv           # Extracted LBP features
├── src/
│   ├── config.py              # Central project configuration & paths
│   ├── preprocessing.py       # Image resizing and normalization
│   ├── glcm_features.py       # GLCM texture extraction
│   ├── lbp_features.py        # Multi-scale LBP texture extraction
│   ├── feature_fusion.py      # Feature concatenation and scaling
│   ├── train_models.py        # Training and baseline evaluation
│   ├── retrain_balanced.py    # Class-weighted balanced training
│   ├── retrain_final.py       # Final model training with SelectKBest
│   ├── evaluation.py          # Metrics, confusion matrix, ROC-AUC
│   └── utils.py               # Logging and plotting helpers
├── notebooks/                 # Jupyter exploration & analysis notebooks (01 to 09)
├── outputs/
│   ├── models/                # Serialized trained models (.pkl)
│   ├── figures/               # Evaluation plots and figures
│   └── results/               # Metric benchmark CSVs
└── splits/                    # Patient-stratified Train / Val / Test splits
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/tabrejansary/ML-Project.git
cd ML-Project
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the Diagnostic Web App

Run the interactive Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Training Models & Running Experiments

- **Train all 6 classifiers**:
  ```bash
  python src/train_models.py
  ```
- **Retrain with balanced weights**:
  ```bash
  python src/retrain_balanced.py
  ```
- **Final retrain with feature selection**:
  ```bash
  python src/retrain_final.py
  ```

---

## 📊 Dataset

This project uses the **BreaKHis** (Breast Cancer Histopathological Image Classification) dataset.
- The pre-computed feature representations (`features_fused.csv`, `features_glcm.csv`, `features_lbp.csv`) and trained models (`outputs/models/`) are already included in this repository, so you can run the web app and evaluation scripts immediately without downloading the gigabyte-scale raw image dataset.
- If you wish to extract features from scratch, download BreaKHis and place it in `dataset/BreaKHis_v1/`.
