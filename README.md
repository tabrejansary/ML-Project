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

## 📊 Dataset Setup (BreaKHis)

This project uses the **BreaKHis** (Breast Cancer Histopathological Image Classification) dataset (~4.1 GB, 7,909 images across 40X, 100X, 200X, and 400X magnifications).

Because of GitHub's repository size limits, the raw image folder (`dataset/`) is excluded from Git. Follow these steps to add the dataset:

### Step 1: Download the Dataset
Download BreaKHis from either of these sources:
- **Kaggle**: [ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis)
  - Quick download using Kaggle CLI:
    ```bash
    kaggle datasets download -d ambarish/breakhis -p dataset/ --unzip
    ```
- **Official Laboratory**: [UFPR BreaKHis Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

### Step 2: Extract & Place the Files
Extract the archive directly into the `dataset/` folder so the directory structure is:

```text
ML-Project/
└── dataset/
    └── BreaKHis_v1/
        └── histology_slides/
            └── breast/
                ├── benign/
                │   └── SOB/
                │       ├── adenosis/
                │       ├── fibroadenoma/
                │       ├── phyllodes_tumor/
                │       └── tubular_adenoma/
                └── malignant/
                    └── SOB/
                        ├── ductal_carcinoma/
                        ├── lobular_carcinoma/
                        ├── mucinous_carcinoma/
                        └── papillary_carcinoma/
```

### Step 3: Run the Full Pipeline with the Dataset
Once the dataset is placed inside `dataset/BreaKHis_v1/`:
1. **Explore the dataset & visualize slides**:
   Open and run [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb).
2. **Re-extract GLCM and LBP features**:
   ```bash
   python src/extract_features.py
   ```
3. **Train classifiers or run the app**:
   ```bash
   python src/train_models.py
   streamlit run app.py
   ```

> **Note:** The extracted feature CSVs (`features_fused.csv`, `features_glcm.csv`, `features_lbp.csv`, `metadata.csv`) and trained model weights (`outputs/models/`) are already provided in this repository, so the web application and evaluation benchmarks can also run immediately even before downloading the full 4.1 GB raw image archive.
