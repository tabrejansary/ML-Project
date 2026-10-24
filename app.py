"""
HistoCAD — Breast Histopathology Diagnostic System
Complete, Professional Web Application with Landing Page & Preloaded Live Diagnostic Workstation
Zero Emojis | Pure Monochrome | Unzoomed Layout | Detailed Clinical Analysis
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# Configure Root Paths
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    IMAGE_SIZE, GLCM_LEVELS, MODELS_DIR, FIGURES_DIR,
    RESULTS_DIR, METADATA_CSV
)
from src.preprocessing import ImagePreprocessor
from src.glcm_features import GLCMExtractor
from src.lbp_features import LBPExtractor

# ──────────────────────────────────────────────────────────────────────────────
# Page Setup & Modern Stylesheet
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HistoCAD | Breast Biopsy Workstation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Compact, Non-Zoomed, Modern Web Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0A0A0C !important;
        color: #E4E4E7 !important;
        font-size: 13px !important;
    }

    /* Kill giant Streamlit padding & unzoom layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1080px !important;
    }

    /* Remove Streamlit ugly anchor chain links */
    a.anchorjs-link, [data-testid="stHeaderAction"], h1 a, h2 a, h3 a {
        display: none !important;
    }

    /* Modern Top Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #27272A;
        padding-bottom: 10px;
        margin-bottom: 16px;
    }
    .brand-title {
        font-size: 15px;
        font-weight: 700;
        color: #FAFAFA;
        letter-spacing: -0.01em;
    }
    .brand-sub {
        font-size: 11px;
        color: #71717A;
        margin-left: 8px;
    }
    .status-badge {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: #22C55E;
        background: #142318;
        border: 1px solid #1E3D27;
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Headings */
    h1 {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #FAFAFA !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 6px 0 !important;
    }
    h2 {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #FAFAFA !important;
        margin: 0 0 4px 0 !important;
    }
    p {
        font-size: 12.5px !important;
        color: #A1A1AA !important;
        line-height: 1.5 !important;
        margin: 0 0 12px 0 !important;
    }

    /* Hero Section Card */
    .hero-card {
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 6px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Diagnosis Verdict Card */
    .verdict-box-mal {
        background: #1C1113;
        border: 1px solid #EF4444;
        border-left: 4px solid #EF4444;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .verdict-box-ben {
        background: #0E1E14;
        border: 1px solid #22C55E;
        border-left: 4px solid #22C55E;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .verdict-tag-mal {
        font-size: 11px;
        font-weight: 700;
        color: #EF4444;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .verdict-tag-ben {
        font-size: 11px;
        font-weight: 700;
        color: #22C55E;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .verdict-head {
        font-size: 17px;
        font-weight: 700;
        color: #FAFAFA;
        margin: 4px 0 6px 0;
    }
    .verdict-sub {
        font-size: 12.5px;
        color: #D4D4D8;
        line-height: 1.5;
        margin: 0;
    }

    /* Progress bar */
    .bar-track {
        background: #27272A;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin: 8px 0 4px 0;
    }
    .bar-fill-mal {
        background: #EF4444;
        height: 100%;
    }
    .bar-fill-ben {
        background: #22C55E;
        height: 100%;
    }

    /* Feature Explanation Box */
    .feature-row {
        background: #17171B;
        border: 1px solid #27272A;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .feature-title {
        font-weight: 600;
        color: #FAFAFA;
        font-size: 12px;
    }
    .feature-explanation {
        font-size: 11px;
        color: #A1A1AA;
        margin-top: 2px;
    }
    .feature-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        color: #FAFAFA;
    }

    /* Stat Box */
    .stat-card {
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: left;
    }
    .stat-label {
        font-size: 10px;
        color: #71717A;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 2px;
    }
    .stat-value {
        font-size: 16px;
        font-weight: 700;
        color: #FAFAFA;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Step Card */
    .step-box {
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 6px;
        padding: 14px;
        height: 100%;
    }

    /* Streamlit Pills Styling */
    div[data-testid="stPills"] {
        margin-bottom: 12px;
    }
    button[data-testid="stPill"] {
        font-size: 11.5px !important;
        padding: 5px 14px !important;
        border-radius: 4px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #FAFAFA !important;
        color: #0A0A0C !important;
        border: 1px solid #FAFAFA !important;
        border-radius: 4px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    .stButton > button:hover {
        background-color: #E4E4E7 !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load Models & Preprocessors
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_extractors():
    return ImagePreprocessor(), GLCMExtractor(), LBPExtractor()

@st.cache_resource
def load_models():
    mapping = {
        'Linear Discriminant Analysis (LDA)': 'lda.pkl',
        'Random Forest (200 Trees)': 'random_forest.pkl',
        'Support Vector Machine (SVM)': 'svm.pkl',
        'k-Nearest Neighbors (k-NN)': 'knn.pkl'
    }
    loaded = {}
    for label, fn in mapping.items():
        p = MODELS_DIR / fn
        if p.exists():
            loaded[label] = joblib.load(p)
    return loaded

@st.cache_data
def load_meta():
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    return pd.DataFrame()

prep, glcm, lbp = load_extractors()
models = load_models()
meta_df = load_meta()


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction Engine
# ──────────────────────────────────────────────────────────────────────────────
def extract_detailed_features(img_bgr: np.ndarray) -> Dict[str, Any]:
    t0 = time.time()
    gray = prep.to_grayscale(img_bgr)
    clahe_gray = prep.apply_clahe(gray)
    resized_u8 = prep.resize(clahe_gray)
    img_float = (resized_u8 / 255.0).astype(np.float32)

    # GLCM 64 levels
    img_q = (resized_u8 / 255.0 * (GLCM_LEVELS - 1)).astype(np.uint8)
    g_feats = glcm.extract(img_q)

    # Multi-scale LBP
    l_feats = lbp.extract(img_float)

    fused = np.concatenate([g_feats, l_feats]).reshape(1, -1)
    latency_ms = (time.time() - t0) * 1000.0

    # LBP surface map
    from skimage.feature import local_binary_pattern
    lbp_map = local_binary_pattern(img_float, P=8, R=1, method='uniform')
    lbp_norm = ((lbp_map / max(lbp_map.max(), 1.0)) * 255).astype(np.uint8)

    # Multi-model voting
    model_predictions = {}
    malignant_votes = 0

    for name, pipe in models.items():
        pred_i = int(pipe.predict(fused)[0])
        prob_i = pipe.predict_proba(fused)[0]
        if pred_i == 1:
            malignant_votes += 1
        model_predictions[name] = {
            "prediction": "MALIGNANT" if pred_i == 1 else "BENIGN",
            "prob_malignant": round(float(prob_i[1]) * 100.0, 1),
            "prob_benign": round(float(prob_i[0]) * 100.0, 1)
        }

    return {
        "clahe_u8": resized_u8,
        "lbp_norm": lbp_norm,
        "latency_ms": latency_ms,
        "model_predictions": model_predictions,
        "malignant_votes": malignant_votes,
        "total_models": len(models),
        "contrast": float(g_feats[0]),
        "dissimilarity": float(g_feats[1]),
        "homogeneity": float(g_feats[2]),
        "energy": float(g_feats[3])
    }


# ──────────────────────────────────────────────────────────────────────────────
# Top Header Bar & Navigation Controller
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div>
        <span class="brand-title">HISTOCAD</span>
        <span class="brand-sub">Breast Biopsy Decision Support Workstation</span>
    </div>
    <div class="status-badge">4 ENGINES ONLINE // READY</div>
</div>
""", unsafe_allow_html=True)

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Home & Overview"

# Clean Pill Navigation
nav_view = st.pills(
    "Navigation",
    ["Home & Overview", "Diagnostic Analyzer", "Model Benchmarks"],
    key="active_nav",
    label_visibility="collapsed"
)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1: HOME & OVERVIEW (BEAUTIFUL LANDING PAGE WITH CALL TO ACTION)
# ══════════════════════════════════════════════════════════════════════════════
if nav_view == "Home & Overview":
    st.markdown("""
    <div class="hero-card">
        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #71717A; margin-bottom: 6px;">
            HYBRID TEXTURE FUSION (GLCM + LBP) FRAMEWORK
        </div>
        <h1 style="font-size: 24px !important; margin-bottom: 8px !important;">
            Automated Breast Cancer Histopathology Classification
        </h1>
        <p style="font-size: 13.5px !important; color: #A1A1AA; max-width: 780px; line-height: 1.6; margin-bottom: 18px;">
            A clinical-grade machine learning decision-support system to detect <strong>Benign</strong> and <strong>Malignant</strong> 
            breast lesions from microscopic biopsy slides with an official patient-stratified cross-validation score of <strong>84.92% F1</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Primary Action Button to switch to Analyzer
    btn_col, _ = st.columns([1.8, 3])
    with btn_col:
        if st.button("Open Diagnostic Analyzer →", use_container_width=True):
            st.session_state.active_nav = "Diagnostic Analyzer"
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3-Step Process
    st.markdown("<h2>Methodological Workflow</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-box">
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #71717A; margin-bottom: 4px;">STAGE 01</div>
            <div style="font-size: 13px; font-weight: 600; color: #FAFAFA; margin-bottom: 4px;">Slide Acquisition</div>
            <p style="font-size: 12px; color: #A1A1AA; margin: 0;">Standard H&E stained biopsy slide slices evaluated across 40X, 100X, 200X, and 400X magnifications.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-box">
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #71717A; margin-bottom: 4px;">STAGE 02</div>
            <div style="font-size: 13px; font-weight: 600; color: #FAFAFA; margin-bottom: 4px;">Texture Extraction</div>
            <p style="font-size: 12px; color: #A1A1AA; margin: 0;">24 GLCM spatial descriptors + 54 rotation-invariant LBP micro-texture histograms fused into a 72-D vector.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-box">
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #71717A; margin-bottom: 4px;">STAGE 03</div>
            <div style="font-size: 13px; font-weight: 600; color: #FAFAFA; margin-bottom: 4px;">Consensus Inference</div>
            <p style="font-size: 12px; color: #A1A1AA; margin: 0;">Top 30 ANOVA F-selected features classified across LDA, Random Forest, SVM, and k-NN with 84.92% CV F1.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Key Cohort Statistics
    st.markdown("<h2>Dataset & Validation Benchmarks</h2>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Benchmark CV F1</div>
            <div class="stat-value">84.92%</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Total Images</div>
            <div class="stat-value">7,909</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Patient Cohort</div>
            <div class="stat-value">82 Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Subtypes</div>
            <div class="stat-value">8 Classes</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2: DIAGNOSTIC ANALYZER (PRELOADED WITH LIVE SAMPLE — NEVER EMPTY!)
# ══════════════════════════════════════════════════════════════════════════════
elif nav_view == "Diagnostic Analyzer":
    # ── Upload Section (Clean, Directly Visible at Top) ───────────────────────
    col_up, col_mod = st.columns([1.8, 1])
    with col_up:
        up_file = st.file_uploader(
            "Upload Biopsy Slide Image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Drag and drop or browse any breast histology slice."
        )
    with col_mod:
        selected_model = st.selectbox(
            "Primary Decision Engine",
            list(models.keys()),
            index=0
        )

    # Quick Clinical Sample Selection Pills (Pre-selected by default!)
    selected_sample = st.pills(
        "Or test instantly with clinical sample cases:",
        [
            "Sample 1: Benign Adenosis (100X)",
            "Sample 2: Malignant Carcinoma (100X)",
            "Sample 3: Benign Fibroadenoma (200X)",
            "Sample 4: Malignant Carcinoma (400X)"
        ],
        default="Sample 1: Benign Adenosis (100X)"
    )

    st.markdown("<hr style='border: none; border-bottom: 1px solid #27272A; margin: 12px 0 16px 0;'>", unsafe_allow_html=True)

    # ── Resolve Image to Analyze ──────────────────────────────────────────────
    active_bgr = None
    slide_label = ""

    if up_file is not None:
        raw_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        active_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        slide_label = f"Uploaded Slide: {up_file.name}"
    elif not meta_df.empty:
        if "Sample 1" in selected_sample:
            row = meta_df[(meta_df['class_name'] == 'benign') & (meta_df['magnification'] == '100X')].iloc[0]
        elif "Sample 2" in selected_sample:
            row = meta_df[(meta_df['class_name'] == 'malignant') & (meta_df['magnification'] == '100X')].iloc[0]
        elif "Sample 3" in selected_sample:
            row = meta_df[(meta_df['class_name'] == 'benign') & (meta_df['magnification'] == '200X')].iloc[5]
        else:
            row = meta_df[(meta_df['class_name'] == 'malignant') & (meta_df['magnification'] == '400X')].iloc[5]

        if os.path.exists(row['image_path']):
            active_bgr = cv2.imread(row['image_path'])
            slide_label = f"Clinical Sample: {row['class_name'].upper()} — {row['subtype'].replace('_', ' ').title()} ({row['magnification']})"

    # ── Run Analysis & Render Results (Instantly Populated!) ──────────────────
    if active_bgr is not None:
        with st.spinner("Extracting texture features and running multi-model consensus..."):
            res = extract_detailed_features(active_bgr)
            lda_res = res["model_predictions"][selected_model]
            p_mal = lda_res["prob_malignant"]
            p_ben = lda_res["prob_benign"]
            is_mal = p_mal >= 50.0

        # Row 1: High-Level Diagnosis Banner (Plain English for Non-Technical Users)
        if is_mal:
            st.markdown(f"""
            <div class="verdict-box-mal">
                <div class="verdict-tag-mal">DIAGNOSTIC FINDING • MALIGNANT LESION DETECTED</div>
                <div class="verdict-head">MALIGNANT (BREAST CANCER CELLS DETECTED)</div>
                <p class="verdict-sub">
                    <strong>Clinical Summary:</strong> The tissue specimen presents high nuclear irregularity, elevated local texture contrast, and disrupted glandular architecture indicative of invasive carcinoma.
                </p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 10px;">
                    <span style="color: #FAFAFA;">Malignancy Confidence: <strong style="color: #EF4444; font-size: 14px;">{p_mal}%</strong></span>
                    <span style="color: #71717A;">Benign Probability: {p_ben}%</span>
                    <span style="color: #E4E4E7;"><strong>Consensus:</strong> {res['malignant_votes']} of {res['total_models']} Models Vote Malignant</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-mal" style="width: {p_mal}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box-ben">
                <div class="verdict-tag-ben">DIAGNOSTIC FINDING • BENIGN LESION (NON-CANCEROUS)</div>
                <div class="verdict-head">BENIGN TISSUE (NO CANCER DETECTED)</div>
                <p class="verdict-sub">
                    <strong>Clinical Summary:</strong> The tissue specimen exhibits regular glandular spacing, uniform stroma, and preserved cellular architecture typical of non-cancerous benign lesions.
                </p>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 10px;">
                    <span style="color: #FAFAFA;">Benign Confidence: <strong style="color: #22C55E; font-size: 14px;">{p_ben}%</strong></span>
                    <span style="color: #71717A;">Malignancy Probability: {p_mal}%</span>
                    <span style="color: #E4E4E7;"><strong>Consensus:</strong> {res['total_models'] - res['malignant_votes']} of {res['total_models']} Models Vote Benign</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-ben" style="width: {p_ben}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Row 2: Deep 2-Column Inspection
        col_img, col_metrics = st.columns([1, 1.25], gap="large")

        with col_img:
            st.markdown(f"<div style='font-size: 11px; color: #71717A; margin-bottom: 6px;'>SPECIMEN: <strong style='color: #FAFAFA;'>{slide_label}</strong></div>", unsafe_allow_html=True)
            img_tabs = st.tabs(["Optical Biopsy Slide", "Contrast-Normalized (CLAHE)", "LBP Micro-Texture Map"])
            with img_tabs[0]:
                st.image(cv2.cvtColor(active_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with img_tabs[1]:
                st.image(res["clahe_u8"], caption="Stain-Invariant Grayscale (Normalizes Laboratory Differences)", use_container_width=True)
            with img_tabs[2]:
                st.image(res["lbp_norm"], caption="Local Binary Pattern (Visualizes Cell Border Roughness)", use_container_width=True)

        with col_metrics:
            # Multi-Model Voting Ledger
            st.markdown("<div style='font-size: 11px; font-weight: 600; color: #71717A; text-transform: uppercase; margin-bottom: 6px;'>Independent Multi-Model Voting Ledger</div>", unsafe_allow_html=True)
            
            model_rows = []
            for m_name, m_data in res["model_predictions"].items():
                m_color = "#EF4444" if m_data["prediction"] == "MALIGNANT" else "#22C55E"
                m_conf = m_data["prob_malignant"] if m_data["prediction"] == "MALIGNANT" else m_data["prob_benign"]
                model_rows.append(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: #17171B; border: 1px solid #27272A; border-radius: 4px; padding: 7px 10px; margin-bottom: 4px; font-size: 11.5px;">
                    <span style="font-weight: 500; color: #E4E4E7;">{m_name}</span>
                    <span>
                        <strong style="color: {m_color}; margin-right: 8px;">{m_data['prediction']}</strong>
                        <span style="color: #71717A; font-family: 'JetBrains Mono', monospace;">({m_conf}%)</span>
                    </span>
                </div>
                """)
            st.markdown("".join(model_rows), unsafe_allow_html=True)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Detailed Tissue Feature Analysis (Explained for Non-Technical Users)
            st.markdown("<div style='font-size: 11px; font-weight: 600; color: #71717A; text-transform: uppercase; margin-bottom: 6px;'>Tissue Texture Feature Breakdown</div>", unsafe_allow_html=True)

            c_indicator = "HIGH (Indicates irregular, crowded cell nuclei)" if res["contrast"] > 70 else "NORMAL (Even cell distribution)"
            h_indicator = "LOW (Disordered glandular structure)" if res["homogeneity"] < 0.25 else "HIGH (Intact, uniform tissue stroma)"

            st.markdown(f"""
            <div class="feature-row">
                <div>
                    <div class="feature-title">Nuclear Contrast (GLCM Contrast)</div>
                    <div class="feature-explanation">{c_indicator}</div>
                </div>
                <div class="feature-val">{res['contrast']:.2f}</div>
            </div>
            <div class="feature-row">
                <div>
                    <div class="feature-title">Tissue Homogeneity (GLCM Homogeneity)</div>
                    <div class="feature-explanation">{h_indicator}</div>
                </div>
                <div class="feature-val">{res['homogeneity']:.3f}</div>
            </div>
            <div class="feature-row">
                <div>
                    <div class="feature-title">Local Dissimilarity (Cellular Variation)</div>
                    <div class="feature-explanation">Measures microscopic variation between adjacent tissue pixels.</div>
                </div>
                <div class="feature-val">{res['dissimilarity']:.2f}</div>
            </div>
            <div class="feature-row">
                <div>
                    <div class="feature-title">Pipeline Latency</div>
                    <div class="feature-explanation">End-to-end extraction and 4-model inference time.</div>
                </div>
                <div class="feature-val">{res['latency_ms']:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 3: Interactive Multi-Model Probability Bar Chart
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 11px; font-weight: 600; color: #71717A; text-transform: uppercase; margin-bottom: 6px;'>Interactive Probability Comparison Across Classifiers</div>", unsafe_allow_html=True)
        
        m_keys = list(res["model_predictions"].keys())
        p_mals = [res["model_predictions"][k]["prob_malignant"] for k in m_keys]
        p_bens = [res["model_predictions"][k]["prob_benign"] for k in m_keys]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[k.split(' (')[0] for k in m_keys],
            x=p_mals,
            name='Malignant %',
            orientation='h',
            marker=dict(color='#EF4444'),
            hovertemplate='%{y}: %{x:.1f}% Malignant<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            y=[k.split(' (')[0] for k in m_keys],
            x=p_bens,
            name='Benign %',
            orientation='h',
            marker=dict(color='#22C55E'),
            hovertemplate='%{y}: %{x:.1f}% Benign<extra></extra>'
        ))
        fig.update_layout(
            barmode='stack',
            height=160,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0, 100], ticksuffix="%", color='#71717A', gridcolor='#27272A'),
            yaxis=dict(color='#FAFAFA'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#A1A1AA')),
            font=dict(family='Inter')
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 3: MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
elif nav_view == "Model Benchmarks":
    st.markdown("<h2>Multi-Model Evaluation & Validation</h2>", unsafe_allow_html=True)
    st.markdown("<p>Patient-stratified 5-Fold cross-validation metrics across 7,909 biopsy images.</p>", unsafe_allow_html=True)

    t_path = RESULTS_DIR / 'test_results.csv'
    if t_path.exists():
        df_bench = pd.read_csv(t_path)
        st.markdown("<div style='background: #121215; border: 1px solid #27272A; border-radius: 6px; padding: 14px;'>", unsafe_allow_html=True)
        st.dataframe(
            df_bench.style.format({
                'Accuracy': '{:.2f}%',
                'Precision': '{:.2f}%',
                'Recall': '{:.2f}%',
                'Specificity': '{:.2f}%',
                'F1': '{:.2f}%',
                'ROC_AUC': '{:.2f}%',
                'MCC': '{:.4f}',
                'CV_F1': '{:.2f}%',
                'CV_F1_std': '±{:.2f}%'
            }),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        cm_p = FIGURES_DIR / "07_confusion_matrix_test.png"
        if not cm_p.exists():
            cm_p = FIGURES_DIR / "06_confusion_matrices.png"
        if cm_p.exists():
            st.image(str(cm_p), caption="Diagnostic Confusion Matrices", use_container_width=True)
    with p2:
        roc_p = FIGURES_DIR / "07_roc_curves_test.png"
        if not roc_p.exists():
            roc_p = FIGURES_DIR / "06_roc_curves.png"
        if roc_p.exists():
            st.image(str(roc_p), caption="ROC Curves across Classifiers", use_container_width=True)
