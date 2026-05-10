import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import json
import importlib.util
import sys
from pathlib import Path
import zipfile
import tempfile
import os
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
from collections import Counter
import itertools

# === Page Config ===
st.set_page_config(
    page_title="Carelens AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS ===
st.markdown("""
<style>
    /* Navbar styling */
    .nav-container {
        display: flex;
        gap: 8px;
        padding: 12px 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 24px;
    }
    .stRadio > div {
        display: flex;
        flex-direction: row;
        gap: 10px;
    }
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #0066cc;
        margin: 8px 0;
    }
    /* Section headers */
    .section-tag {
        background: #0066cc;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .section-tag-green {
        background: #28a745;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# === Dynamically import medifusion_V1.0.py ===
module_path = Path(__file__).parent / "medifusion_V1.0.py"
spec = importlib.util.spec_from_file_location("medifusion_v1", str(module_path))
medifusion_v1 = importlib.util.module_from_spec(spec)
sys.modules["medifusion_v1"] = medifusion_v1
spec.loader.exec_module(medifusion_v1)

# === Assign the classes ===
SEBlock = medifusion_v1.SEBlock
BayesianDense = medifusion_v1.BayesianDense
VisionTransformerBlock = medifusion_v1.VisionTransformerBlock
AdaptiveMultiScaleAttention = medifusion_v1.AdaptiveMultiScaleAttention
AnatomicalReasoningModule = medifusion_v1.AnatomicalReasoningModule
AdaptiveFocalLoss = medifusion_v1.AdaptiveFocalLoss

# === Load model ===
@st.cache_resource
def load_medifusion_model(model_path):
    custom_objects = {
        "SEBlock": SEBlock,
        "BayesianDense": BayesianDense,
        "VisionTransformerBlock": VisionTransformerBlock,
        "AdaptiveMultiScaleAttention": AdaptiveMultiScaleAttention,
        "AnatomicalReasoningModule": AnatomicalReasoningModule,
        "AdaptiveFocalLoss": AdaptiveFocalLoss,
    }
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    return model

model = load_medifusion_model("medifusion_enhanced_best.keras")

# === Load class indices ===
with open("medifusion_class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# ============================================================
# SHARED UTILITY FUNCTIONS
# ============================================================

def is_image_file(filename: str) -> bool:
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in valid_extensions

def extract_images_from_zip(zip_file) -> List[tuple]:
    extracted_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir() and is_image_file(file_info.filename):
                    try:
                        zip_ref.extract(file_info, temp_dir)
                        extracted_path = os.path.join(temp_dir, file_info.filename)
                        img = Image.open(extracted_path).convert("RGB")
                        extracted_files.append((file_info.filename, img))
                    except Exception as e:
                        st.warning(f"Could not process {file_info.filename}: {str(e)}")
    return extracted_files

def preprocess_image(img: Image.Image) -> np.ndarray:
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_single_image(img: Image.Image) -> Dict[str, Any]:
    img_batch = preprocess_image(img)
    predictions = model.predict(img_batch, verbose=0)
    main_preds = predictions['main_predictions'][0]
    epistemic_unc = predictions['epistemic_uncertainty'][0][0]
    aleatoric_unc = predictions['aleatoric_uncertainty'][0][0]
    pred_idx = np.argmax(main_preds)
    pred_class = index_to_class[pred_idx]
    confidence = main_preds[pred_idx]
    return {
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'epistemic_uncertainty': float(epistemic_unc),
        'aleatoric_uncertainty': float(aleatoric_unc),
        'prediction_index': pred_idx,
        'all_predictions': main_preds,
        'processed_image': img_batch
    }

def generate_gradcam(img_batch: np.ndarray, pred_idx: int) -> np.ndarray:
    last_conv_layer_name = "conv2d_21"
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output['main_predictions']]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            loss = predictions[:, pred_idx]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0 or tf.math.is_nan(max_val):
            return None
        heatmap = heatmap / (max_val + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy().astype(np.float32)
        if np.isnan(heatmap).any() or heatmap.ndim != 2:
            return None
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        img_array = img_batch[0]
        superimposed = cv2.addWeighted(np.uint8(img_array * 255), 0.6, heatmap_colored, 0.4, 0)
        return superimposed
    except Exception as e:
        st.error(f"Failed to generate Grad-CAM: {str(e)}")
        return None

def display_single_prediction(filename: str, img: Image.Image, result: Dict[str, Any], show_gradcam: bool = True):
    st.subheader(f"📋 Results for: {filename}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=f"Original: {filename}", use_column_width=True)
    with col2:
        st.write(f"**🎯 Predicted Class:** {result['predicted_class']}")
        st.write(f"**📊 Confidence:** {result['confidence']:.4f}")
        st.write(f"**🔍 Epistemic Uncertainty:** {result['epistemic_uncertainty']:.4f}")
        st.write(f"**⚡ Aleatoric Uncertainty:** {result['aleatoric_uncertainty']:.4f}")
        st.write("**📈 Confidence Distribution:**")
        pred_df = pd.DataFrame({
            'Class': [index_to_class[i] for i in range(len(result['all_predictions']))],
            'Confidence': result['all_predictions']
        })
        st.bar_chart(pred_df.set_index('Class'))
    if show_gradcam:
        st.subheader("🔥 Class Activation Map (Grad-CAM)")
        gradcam = generate_gradcam(result['processed_image'], result['prediction_index'])
        if gradcam is not None:
            st.image(gradcam, caption="Grad-CAM Visualization", use_column_width=True)
        else:
            st.warning("⚠️ Could not generate Grad-CAM for this image")
    st.markdown("---")

def create_batch_summary(results: List[Dict[str, Any]], filenames: List[str]) -> pd.DataFrame:
    summary_data = []
    for filename, result in zip(filenames, results):
        summary_data.append({
            'Filename': filename,
            'Predicted Class': result['predicted_class'],
            'Confidence': result['confidence'],
            'Epistemic Uncertainty': result['epistemic_uncertainty'],
            'Aleatoric Uncertainty': result['aleatoric_uncertainty']
        })
    return pd.DataFrame(summary_data)

def fig_to_bytes(fig) -> bytes:
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# ============================================================
# SECTION A — ANALYSIS FUNCTIONS
# ============================================================

def compute_statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Unit 2: Measures of central tendency and dispersion"""
    stats = {}
    for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
        vals = df[col].astype(float).values
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        stats[col] = {
            'mean':   float(np.mean(vals)),
            'median': float(np.median(vals)),
            'mode':   float(pd.Series(vals).round(2).mode()[0]) if len(vals) > 0 else 0,
            'std':    float(np.std(vals)),
            'variance': float(np.var(vals)),
            'min':    float(np.min(vals)),
            'max':    float(np.max(vals)),
            'range':  float(np.max(vals) - np.min(vals)),
            'q1':     float(q1),
            'q3':     float(q3),
            'iqr':    float(q3 - q1),
        }
    return stats

def plot_stat_distributions(df: pd.DataFrame):
    """Unit 2: Visualize distributions of confidence and uncertainty"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Statistical Distributions of Prediction Metrics", fontsize=14, fontweight='bold')

    metrics = ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']
    colors = ['#0066cc', '#e85d04', '#2d6a4f']

    for ax, metric, color in zip(axes, metrics, colors):
        vals = df[metric].astype(float).values
        ax.hist(vals, bins=min(15, len(vals)), color=color, alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(vals), color='black', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(vals):.3f}')
        ax.axvline(np.median(vals), color='red', linestyle=':', linewidth=1.5, label=f'Median={np.median(vals):.3f}')
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_boxplots(df: pd.DataFrame):
    """Unit 2: Box plots showing IQR, quartiles, outliers"""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']
    data_to_plot = [df[m].astype(float).values for m in metrics]
    bp = ax.boxplot(data_to_plot, labels=metrics, patch_artist=True, notch=False)
    colors = ['#0066cc', '#e85d04', '#2d6a4f']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Box Plot: Quartiles, IQR and Outliers per Metric", fontsize=13, fontweight='bold')
    ax.set_ylabel("Value")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_class_distribution(df: pd.DataFrame):
    """Unit 2: Class distribution bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    class_counts = df['Predicted Class'].value_counts()

    # Bar chart
    axes[0].bar(class_counts.index, class_counts.values,
                color=['#0066cc', '#e85d04', '#2d6a4f'][:len(class_counts)], edgecolor='white')
    axes[0].set_title("Class Distribution (Frequency)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                colors=['#0066cc', '#e85d04', '#2d6a4f'][:len(class_counts)],
                startangle=140, pctdistance=0.85)
    axes[1].set_title("Class Distribution (Proportion)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def compute_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unit 4: Pattern & Association Analysis
    Discretize confidence and uncertainty into bins, then find
    frequent co-occurring patterns across the batch.
    """
    temp = df.copy()

    # Discretize confidence into bins (attributes)
    temp['Conf_Level'] = pd.cut(
        temp['Confidence'].astype(float),
        bins=[0, 0.5, 0.75, 0.9, 1.0],
        labels=['Low(<0.5)', 'Medium(0.5-0.75)', 'High(0.75-0.9)', 'VeryHigh(>0.9)']
    )

    # Discretize epistemic uncertainty
    temp['Epis_Level'] = pd.cut(
        temp['Epistemic Uncertainty'].astype(float),
        bins=[0, 0.1, 0.3, 0.5, 1.0],
        labels=['VeryLow', 'Low', 'Medium', 'High']
    )

    # Discretize aleatoric uncertainty
    temp['Alea_Level'] = pd.cut(
        temp['Aleatoric Uncertainty'].astype(float),
        bins=[0, 0.1, 0.3, 0.5, 1.0],
        labels=['VeryLow', 'Low', 'Medium', 'High']
    )

    rules = []
    total = len(temp)

    # Generate association patterns: Class → Confidence level
    for cls in temp['Predicted Class'].unique():
        cls_mask = temp['Predicted Class'] == cls
        cls_support = cls_mask.sum() / total
        for conf_level in temp['Conf_Level'].dropna().unique():
            conf_mask = temp['Conf_Level'] == conf_level
            both = (cls_mask & conf_mask).sum()
            if both > 0:
                support = both / total
                confidence_rule = both / cls_mask.sum() if cls_mask.sum() > 0 else 0
                lift = confidence_rule / (conf_mask.sum() / total) if (conf_mask.sum() / total) > 0 else 0
                if support >= 0.1:
                    rules.append({
                        'Antecedent': f'Class = {cls}',
                        'Consequent': f'Confidence = {conf_level}',
                        'Support': round(support, 3),
                        'Confidence': round(confidence_rule, 3),
                        'Lift': round(lift, 3)
                    })

    # Generate association patterns: Epistemic → Aleatoric
    for epis in temp['Epis_Level'].dropna().unique():
        epis_mask = temp['Epis_Level'] == epis
        for alea in temp['Alea_Level'].dropna().unique():
            alea_mask = temp['Alea_Level'] == alea
            both = (epis_mask & alea_mask).sum()
            if both > 0:
                support = both / total
                confidence_rule = both / epis_mask.sum() if epis_mask.sum() > 0 else 0
                lift = confidence_rule / (alea_mask.sum() / total) if (alea_mask.sum() / total) > 0 else 0
                if support >= 0.1:
                    rules.append({
                        'Antecedent': f'Epistemic = {epis}',
                        'Consequent': f'Aleatoric = {alea}',
                        'Support': round(support, 3),
                        'Confidence': round(confidence_rule, 3),
                        'Lift': round(lift, 3)
                    })

    # Generate: Class → Epistemic uncertainty level
    for cls in temp['Predicted Class'].unique():
        cls_mask = temp['Predicted Class'] == cls
        for epis in temp['Epis_Level'].dropna().unique():
            epis_mask = temp['Epis_Level'] == epis
            both = (cls_mask & epis_mask).sum()
            if both > 0:
                support = both / total
                confidence_rule = both / cls_mask.sum() if cls_mask.sum() > 0 else 0
                lift = confidence_rule / (epis_mask.sum() / total) if (epis_mask.sum() / total) > 0 else 0
                if support >= 0.1:
                    rules.append({
                        'Antecedent': f'Class = {cls}',
                        'Consequent': f'Epistemic = {epis}',
                        'Support': round(support, 3),
                        'Confidence': round(confidence_rule, 3),
                        'Lift': round(lift, 3)
                    })

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values('Lift', ascending=False).drop_duplicates().reset_index(drop=True)
    return rules_df

def compute_frequent_itemsets(df: pd.DataFrame) -> pd.DataFrame:
    """Unit 4: Frequent patterns — which class+confidence combos appear most"""
    temp = df.copy()
    temp['Conf_Bin'] = pd.cut(
        temp['Confidence'].astype(float),
        bins=[0, 0.5, 0.75, 0.9, 1.0],
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    temp['Epis_Bin'] = pd.cut(
        temp['Epistemic Uncertainty'].astype(float),
        bins=[0, 0.1, 0.3, 0.5, 1.0],
        labels=['VeryLow', 'Low', 'Medium', 'High']
    )

    # Count frequent (Class, ConfBin) pairs
    itemsets = temp.groupby(['Predicted Class', 'Conf_Bin']).size().reset_index(name='Count')
    itemsets['Support'] = (itemsets['Count'] / len(temp)).round(3)
    itemsets = itemsets[itemsets['Support'] >= 0.05].sort_values('Support', ascending=False)
    itemsets.columns = ['Class', 'Confidence Level', 'Count', 'Support']
    return itemsets

def plot_similarity_heatmap(df: pd.DataFrame):
    """Unit 2: Similarity/Dissimilarity — correlation between metrics"""
    numeric_df = df[['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']].astype(float)
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
        vmin=-1, vmax=1, ax=ax, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Metric Correlation (Similarity/Dissimilarity)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_confidence_per_class(df: pd.DataFrame):
    """Unit 2: Mean confidence and uncertainty per class"""
    grouped = df.groupby('Predicted Class').agg(
        Avg_Confidence=('Confidence', 'mean'),
        Avg_Epistemic=('Epistemic Uncertainty', 'mean'),
        Avg_Aleatoric=('Aleatoric Uncertainty', 'mean'),
        Count=('Predicted Class', 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(grouped))
    width = 0.25
    ax.bar(x - width, grouped['Avg_Confidence'], width, label='Avg Confidence', color='#0066cc', alpha=0.85)
    ax.bar(x, grouped['Avg_Epistemic'], width, label='Avg Epistemic Unc.', color='#e85d04', alpha=0.85)
    ax.bar(x + width, grouped['Avg_Aleatoric'], width, label='Avg Aleatoric Unc.', color='#2d6a4f', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['Predicted Class'])
    ax.set_title("Average Metrics per Predicted Class", fontsize=13, fontweight='bold')
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def show_preprocessing_pipeline(img: Image.Image):
    """Unit 3: Show preprocessing steps visually"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Data Preprocessing Pipeline", fontsize=14, fontweight='bold')

    # Step 1: Original
    axes[0].imshow(img)
    axes[0].set_title("Step 1: Original Image", fontsize=10)
    axes[0].axis('off')

    # Step 2: Resized to 224x224
    resized = img.resize((224, 224))
    axes[1].imshow(resized)
    axes[1].set_title("Step 2: Resized (224×224)", fontsize=10)
    axes[1].axis('off')

    # Step 3: Grayscale
    gray = np.array(resized.convert('L'))
    axes[2].imshow(gray, cmap='gray')
    axes[2].set_title("Step 3: Grayscale View", fontsize=10)
    axes[2].axis('off')

    # Step 4: Normalized array
    normalized = np.array(resized) / 255.0
    axes[3].imshow(normalized)
    axes[3].set_title("Step 4: Normalized (÷255)", fontsize=10)
    axes[3].axis('off')

    plt.tight_layout()
    return fig

def generate_html_report(df: pd.DataFrame, stats: Dict, rules_df: pd.DataFrame, itemsets_df: pd.DataFrame) -> str:
    """Unit 1: Generate a downloadable HTML report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(df)
    class_counts = df['Predicted Class'].value_counts().to_dict()
    avg_conf = df['Confidence'].astype(float).mean()
    avg_epis = df['Epistemic Uncertainty'].astype(float).mean()

    class_rows = "".join([
        f"<tr><td>{cls}</td><td>{count}</td><td>{count/total*100:.1f}%</td></tr>"
        for cls, count in class_counts.items()
    ])

    stats_rows = ""
    for metric, vals in stats.items():
        stats_rows += f"""
        <tr>
          <td><b>{metric}</b></td>
          <td>{vals['mean']:.4f}</td>
          <td>{vals['median']:.4f}</td>
          <td>{vals['std']:.4f}</td>
          <td>{vals['variance']:.4f}</td>
          <td>{vals['range']:.4f}</td>
          <td>{vals['q1']:.4f}</td>
          <td>{vals['q3']:.4f}</td>
          <td>{vals['iqr']:.4f}</td>
        </tr>"""

    rules_rows = ""
    if not rules_df.empty:
        for _, row in rules_df.head(10).iterrows():
            rules_rows += f"<tr><td>{row['Antecedent']}</td><td>{row['Consequent']}</td><td>{row['Support']}</td><td>{row['Confidence']}</td><td>{row['Lift']}</td></tr>"

    itemset_rows = ""
    if not itemsets_df.empty:
        for _, row in itemsets_df.iterrows():
            itemset_rows += f"<tr><td>{row['Class']}</td><td>{row['Confidence Level']}</td><td>{row['Count']}</td><td>{row['Support']}</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Carelens AI — Batch Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1a1a2e; background: #f5f7fa; }}
  h1 {{ color: #0066cc; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
  h2 {{ color: #003d99; margin-top: 36px; border-left: 4px solid #0066cc; padding-left: 12px; }}
  h3 {{ color: #444; }}
  .badge {{ background: #0066cc; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }}
  .badge-green {{ background: #28a745; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .summary-card {{ background: white; border-radius: 10px; padding: 16px; border-left: 4px solid #0066cc; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .summary-card .val {{ font-size: 28px; font-weight: 700; color: #0066cc; }}
  .summary-card .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.07); margin-top: 12px; }}
  th {{ background: #0066cc; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4ff; }}
  .footer {{ margin-top: 48px; text-align: center; font-size: 12px; color: #aaa; border-top: 1px solid #ddd; padding-top: 16px; }}
  .info-box {{ background: #e8f0fe; border-radius: 8px; padding: 14px 18px; margin: 12px 0; font-size: 13px; border-left: 4px solid #0066cc; }}
</style>
</head>
<body>
<h1>🩺 Carelens AI — Batch Analysis Report</h1>
<p style="color:#666">Generated: {timestamp} &nbsp;|&nbsp; Total Images Analysed: <b>{total}</b></p>

<div class="info-box">
This report presents a data mining analysis of chest X-ray batch predictions including statistical descriptions (Section A — Unit 2),
data preprocessing details (Unit 3), and frequent pattern &amp; association analysis (Unit 4).
</div>

<h2><span class="badge">Unit 1</span> &nbsp; Data Mining Application Overview</h2>
<div class="summary-grid">
  <div class="summary-card"><div class="val">{total}</div><div class="lbl">Total Images</div></div>
  <div class="summary-card"><div class="val">{len(class_counts)}</div><div class="lbl">Distinct Classes</div></div>
  <div class="summary-card"><div class="val">{avg_conf:.3f}</div><div class="lbl">Avg Confidence</div></div>
  <div class="summary-card"><div class="val">{avg_epis:.3f}</div><div class="lbl">Avg Epistemic Unc.</div></div>
</div>

<h2><span class="badge">Unit 2</span> &nbsp; Statistical Descriptions &amp; Attribute Analysis</h2>
<h3>Class Distribution (Nominal Attribute)</h3>
<table>
  <tr><th>Predicted Class</th><th>Count</th><th>Proportion</th></tr>
  {class_rows}
</table>

<h3>Measures of Central Tendency &amp; Dispersion (Numeric Attributes)</h3>
<table>
  <tr><th>Attribute</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Variance</th><th>Range</th><th>Q1</th><th>Q3</th><th>IQR</th></tr>
  {stats_rows}
</table>

<h2><span class="badge-green">Unit 3</span> &nbsp; Data Preprocessing</h2>
<div class="info-box">
  All X-ray images were preprocessed through the following pipeline before classification:<br><br>
  <b>1.</b> Loaded as RGB (3-channel) images &nbsp;→&nbsp;
  <b>2.</b> Resized to 224×224 pixels &nbsp;→&nbsp;
  <b>3.</b> Pixel values normalized to [0, 1] by dividing by 255 &nbsp;→&nbsp;
  <b>4.</b> Expanded to batch dimension (1, 224, 224, 3) for model input
</div>

<h2><span class="badge-green">Unit 4</span> &nbsp; Frequent Pattern &amp; Association Analysis</h2>
<h3>Frequent Itemsets (Class × Confidence Level)</h3>
<table>
  <tr><th>Class</th><th>Confidence Level</th><th>Count</th><th>Support</th></tr>
  {itemset_rows if itemset_rows else '<tr><td colspan="4" style="text-align:center;color:#999">Not enough data — need more images for frequent itemsets</td></tr>'}
</table>

<h3>Association Rules (Antecedent → Consequent)</h3>
<table>
  <tr><th>Antecedent</th><th>Consequent</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>
  {rules_rows if rules_rows else '<tr><td colspan="5" style="text-align:center;color:#999">Not enough data — upload more images for association rules</td></tr>'}
</table>
<div class="info-box">
  <b>Reading the rules:</b> Support = fraction of cases where both items appear. Confidence = how often the rule is correct.
  Lift &gt; 1 means the association is stronger than random chance.
</div>

<div class="footer">
  🩺 Carelens AI — Advanced Medical Image Analysis &nbsp;|&nbsp; MediFusion Architecture &nbsp;|&nbsp; Built with Streamlit &amp; TensorFlow
</div>
</body>
</html>"""
    return html


# ============================================================
# NAVBAR
# ============================================================

st.sidebar.title("🩺 Carelens AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Classifier",
        "📊 Batch Analysis & Report",
        "🔬 Preprocessing Visualizer",
        "🧮 Similarity & Distance Matrix",
        "⚠️ Outlier & Anomaly Detection",
        "🛡️ Data Quality Checker",
        "🔎 k-NN Similarity Finder",
        "📉 Uncertainty Analysis Dashboard",
        "📐 Bias / Variance Explorer",
    ],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Supported Formats:** PNG, JPG, JPEG, BMP, TIFF

**Model:** MediFusion-Net Enhanced
- Vision Transformer Hybrid
- Bayesian Uncertainty Quantification
- Grad-CAM Explainability

**Sections Covered:**
- 🔵 Section B: Classification (Unit 6)
- 🟢 Section A: Stats (Unit 2), Preprocessing (Unit 3), Patterns (Unit 4), Uncertainty (Unit 5)
""")


# ============================================================
# PAGE 1: CLASSIFIER (original, untouched logic)
# ============================================================

if page == "🏠 Classifier":
    st.title("🩺 Carelens AI — Advanced Medical X-Ray Classifier")
    st.markdown("Upload single images, multiple files, a ZIP folder, or capture with camera for batch processing.")

    st.info("""
    **What this does:**

    This is the core diagnostic tool of Carelens AI. Upload one or more chest X-ray images and the
    **MediFusion-Net Enhanced** model will classify each scan and return:

    - 🎯 **Predicted Class** — The most likely lung condition detected (e.g. Normal, Pneumonia, COVID-19).
    - 📊 **Confidence Score** — How certain the model is about its top prediction (0 to 1).
    - 🔍 **Epistemic Uncertainty** — Model-level uncertainty; high values suggest the scan is unlike anything
      in the training data and may need expert review.
    - ⚡ **Aleatoric Uncertainty** — Data-level uncertainty; high values indicate the image itself is ambiguous
      or noisy, regardless of the model.
    - 🔥 **Grad-CAM Heatmap** — A colour overlay highlighting which regions of the X-ray most influenced the
      prediction, providing visual explainability.

    **Underlying model:** MediFusion-Net uses a hybrid CNN + Vision Transformer backbone with
    Squeeze-and-Excitation blocks, Adaptive Multi-Scale Attention, Anatomical Reasoning, and
    Bayesian uncertainty heads — trained end-to-end with Adaptive Focal Loss.

    **Use case (Section B — Unit 6: Classification):** Each image is independently classified using
    a trained deep learning model. This is supervised classification applied to medical imaging.
    Results can be exported as a CSV and fed into the **Batch Analysis & Report** page for
    further statistical analysis.

    **⚠️ Disclaimer:** This tool is for research and educational purposes only. It is not a substitute
    for a qualified radiologist or clinical diagnosis.
    """)

    st.subheader("📤 Upload Options")
    upload_option = st.radio(
        "Choose your upload method:",
        ["Single Image", "Multiple Images", "ZIP Folder", "Camera Capture"],
        horizontal=True
    )

    uploaded_files = None
    zip_file = None
    camera_image = None

    if upload_option == "Single Image":
        uploaded_files = st.file_uploader(
            "Upload a single X-ray image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=False
        )
        if uploaded_files:
            uploaded_files = [uploaded_files]

    elif upload_option == "Multiple Images":
        uploaded_files = st.file_uploader(
            "Upload multiple X-ray images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True
        )

    elif upload_option == "ZIP Folder":
        zip_file = st.file_uploader(
            "Upload a ZIP file containing X-ray images",
            type=["zip"]
        )

    elif upload_option == "Camera Capture":
        camera_image = st.camera_input("Take a picture")

    if uploaded_files or zip_file or camera_image:
        if camera_image:
            st.info("📸 Processing camera image...")
            try:
                img = Image.open(camera_image).convert("RGB")
                images_to_process = [("camera_capture.jpg", img)]
            except Exception as e:
                st.error(f"❌ Could not process camera image: {str(e)}")
                st.stop()

        elif zip_file:
            st.info("📦 Extracting images from ZIP file...")
            extracted_images = extract_images_from_zip(zip_file)
            if not extracted_images:
                st.error("❌ No valid image files found in the ZIP archive")
                st.stop()
            st.success(f"✅ Found {len(extracted_images)} image(s) in the ZIP file")
            images_to_process = extracted_images

        else:
            images_to_process = []
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    images_to_process.append((uploaded_file.name, img))
                except Exception as e:
                    st.error(f"❌ Could not process {uploaded_file.name}: {str(e)}")

        if images_to_process:
            st.subheader("⚙️ Processing Options")
            col1, col2 = st.columns(2)
            with col1:
                show_individual_results = st.checkbox("Show individual results", value=True)
            with col2:
                show_gradcam = st.checkbox("Generate Grad-CAM visualizations", value=len(images_to_process) <= 5)

            if len(images_to_process) > 10:
                st.warning("⚠️ Processing many images may take some time. Consider disabling Grad-CAM for faster results.")

            if st.button("🚀 Start Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                filenames = []

                for i, (filename, img) in enumerate(images_to_process):
                    status_text.text(f"Processing {filename}... ({i+1}/{len(images_to_process)})")
                    try:
                        result = predict_single_image(img)
                        results.append(result)
                        filenames.append(filename)
                        if show_individual_results:
                            display_single_prediction(filename, img, result, show_gradcam)
                    except Exception as e:
                        st.error(f"❌ Error processing {filename}: {str(e)}")
                    progress_bar.progress((i + 1) / len(images_to_process))

                status_text.text("✅ Processing complete!")

                if len(results) > 1:
                    st.subheader("📊 Batch Processing Summary")
                    summary_df = create_batch_summary(results, filenames)
                    st.dataframe(summary_df, use_container_width=True)

                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv,
                        file_name="medifusion_batch_results.csv",
                        mime="text/csv"
                    )

                    st.subheader("📈 Class Distribution")
                    class_counts = summary_df['Predicted Class'].value_counts()
                    st.bar_chart(class_counts)

                    st.subheader("📊 Confidence Statistics")
                    confidence_values = summary_df['Confidence'].astype(float).tolist()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Confidence", f"{np.mean(confidence_values):.4f}")
                    with col2:
                        st.metric("Min Confidence", f"{np.min(confidence_values):.4f}")
                    with col3:
                        st.metric("Max Confidence", f"{np.max(confidence_values):.4f}")

                    # Save to session state for use in Analysis page
                    st.session_state['last_batch_df'] = summary_df
                    st.info("💡 Go to **Batch Analysis & Report** in the sidebar to run full Section A analysis on these results.")


# ============================================================
# PAGE 2: BATCH ANALYSIS & REPORT
# ============================================================

elif page == "📊 Batch Analysis & Report":
    st.title("📊 Batch Analysis & Report")
    st.markdown("Upload a CSV of previous predictions **or** upload X-ray images to predict and analyse — all Section A topics covered.")

    st.markdown('<span class="section-tag">Section A — Units 1, 2, 3, 4</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    This page performs a full **data mining analysis** on a batch of chest X-ray predictions —
    either generated fresh from uploaded images, or loaded from a previously downloaded CSV.

    **Topics covered:**

    - 📌 **Unit 1 — Data Mining Overview:** Summarises the batch as a data mining task and lets you
      download a complete formatted HTML report covering all sections.
    - 📐 **Unit 2 — Statistical Descriptions:** Computes mean, median, mode, standard deviation,
      variance, range, Q1, Q3, and IQR for confidence and uncertainty scores. Includes distribution
      histograms, box plots, class frequency charts, per-class metric comparisons, and a
      correlation heatmap showing similarity/dissimilarity between numeric attributes.
    - 🧹 **Unit 3 — Data Preprocessing:** Documents the preprocessing pipeline applied to each image
      before classification (RGB conversion → resize to 224×224 → normalise to [0,1] → batch expand).
    - 🔗 **Unit 4 — Frequent Patterns & Association Rules:** Discretises confidence and uncertainty
      into bins, then mines frequent itemsets (Class × Confidence Level) and generates association
      rules with Support, Confidence, and Lift metrics. Lift > 1 indicates a meaningful association.

    **Use case:** After running the Classifier on a batch, export the CSV and load it here to
    produce a complete academic-style analysis report suitable for a data mining coursework submission.
    """)

    # Input method
    input_method = st.radio(
        "Choose data source:",
        ["Upload Images (predict + analyse)", "Upload CSV (from previous run)"],
        horizontal=True
    )

    analysis_df = None

    if input_method == "Upload Images (predict + analyse)":
        uploaded = st.file_uploader(
            "Upload X-ray images for batch analysis",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="analysis_uploader"
        )
        zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="analysis_zip")

        images_to_process = []
        if uploaded:
            for f in uploaded:
                try:
                    img = Image.open(f).convert("RGB")
                    images_to_process.append((f.name, img))
                except:
                    pass
        elif zip_up:
            images_to_process = extract_images_from_zip(zip_up)

        if images_to_process:
            if st.button("🔍 Run Predictions & Analyse", type="primary"):
                progress_bar = st.progress(0)
                results, filenames = [], []
                for i, (fname, img) in enumerate(images_to_process):
                    try:
                        result = predict_single_image(img)
                        results.append(result)
                        filenames.append(fname)
                    except Exception as e:
                        st.warning(f"Skipped {fname}: {e}")
                    progress_bar.progress((i + 1) / len(images_to_process))

                analysis_df = create_batch_summary(results, filenames)
                st.session_state['analysis_df'] = analysis_df
                st.success(f"✅ Predicted {len(results)} images. Analysis ready below.")

        # Restore from session state if available
        if analysis_df is None and 'analysis_df' in st.session_state:
            analysis_df = st.session_state['analysis_df']

    else:
        csv_file = st.file_uploader("Upload CSV (from Classifier page download)", type=["csv"], key="csv_uploader")
        if csv_file:
            analysis_df = pd.read_csv(csv_file)
            # Ensure numeric columns
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                if col in analysis_df.columns:
                    analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
            st.success(f"✅ Loaded {len(analysis_df)} records from CSV.")

        # Also check session state from classifier page
        if analysis_df is None and 'last_batch_df' in st.session_state:
            st.info("💡 Using results from your last Classifier run.")
            analysis_df = st.session_state['last_batch_df']
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                if col in analysis_df.columns:
                    analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')

    # ---- ANALYSIS SECTION ----
    if analysis_df is not None and len(analysis_df) > 0:

        st.markdown("---")
        st.subheader("📋 Loaded Dataset")
        st.dataframe(analysis_df, use_container_width=True)
        st.caption(f"Total records: {len(analysis_df)} | Classes: {analysis_df['Predicted Class'].nunique()}")

        # ---- UNIT 2: STATISTICAL SUMMARY ----
        st.markdown("---")
        st.markdown('<span class="section-tag">Unit 2 — Statistical Descriptions</span>', unsafe_allow_html=True)
        st.subheader("📐 Measures of Central Tendency & Dispersion")

        stats = compute_statistical_summary(analysis_df)

        for metric, vals in stats.items():
            with st.expander(f"📌 {metric}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{vals['mean']:.4f}")
                c2.metric("Median", f"{vals['median']:.4f}")
                c3.metric("Std Dev", f"{vals['std']:.4f}")
                c4.metric("Variance", f"{vals['variance']:.4f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Range", f"{vals['range']:.4f}")
                c2.metric("Q1", f"{vals['q1']:.4f}")
                c3.metric("Q3", f"{vals['q3']:.4f}")
                c4.metric("IQR", f"{vals['iqr']:.4f}")

        st.subheader("📊 Distribution Plots")
        fig1 = plot_stat_distributions(analysis_df)
        st.pyplot(fig1)
        st.download_button("💾 Download Distribution Plot", fig_to_bytes(fig1),
                           file_name="distribution_plot.png", mime="image/png")

        st.subheader("📦 Box Plots (Quartiles & IQR)")
        fig2 = plot_boxplots(analysis_df)
        st.pyplot(fig2)
        st.download_button("💾 Download Box Plot", fig_to_bytes(fig2),
                           file_name="boxplot.png", mime="image/png")

        st.subheader("🗂️ Class Distribution")
        fig3 = plot_class_distribution(analysis_df)
        st.pyplot(fig3)

        st.subheader("📊 Per-Class Metric Comparison")
        fig4 = plot_confidence_per_class(analysis_df)
        st.pyplot(fig4)

        st.subheader("🔗 Metric Similarity / Dissimilarity (Correlation)")
        fig5 = plot_similarity_heatmap(analysis_df)
        st.pyplot(fig5)
        st.caption("Values near +1 = highly similar, near -1 = highly dissimilar, near 0 = no relationship.")

        # ---- UNIT 4: PATTERNS & ASSOCIATIONS ----
        st.markdown("---")
        st.markdown('<span class="section-tag">Unit 4 — Frequent Patterns & Association Rules</span>', unsafe_allow_html=True)
        st.subheader("🔍 Frequent Itemsets (Class × Confidence Level)")

        itemsets_df = compute_frequent_itemsets(analysis_df)
        if not itemsets_df.empty:
            st.dataframe(itemsets_df, use_container_width=True)
            st.caption("Support = fraction of total images matching this pattern. Threshold: 5%.")
        else:
            st.info("Upload more images (≥5 recommended) to find meaningful frequent itemsets.")

        st.subheader("📐 Association Rules")
        rules_df = compute_association_rules(analysis_df)
        if not rules_df.empty:
            st.dataframe(rules_df, use_container_width=True)
            st.markdown("""
            **How to read:**
            - **Support** — How often the antecedent & consequent appear together
            - **Confidence** — Given the antecedent, how often does the consequent hold?
            - **Lift > 1** — The rule is stronger than chance (the more above 1, the better)
            """)

            # Highlight top rule
            top = rules_df.iloc[0]
            st.success(f"🏆 Strongest rule: **{top['Antecedent']} → {top['Consequent']}** "
                       f"(Support={top['Support']}, Confidence={top['Confidence']}, Lift={top['Lift']})")
        else:
            st.info("Upload more images to generate association rules. Minimum support threshold is 10%.")

        # ---- UNIT 1: FULL REPORT DOWNLOAD ----
        st.markdown("---")
        st.markdown('<span class="section-tag">Unit 1 — Data Mining Application Report</span>', unsafe_allow_html=True)
        st.subheader("📄 Generate Full Report")
        st.markdown("Download a complete HTML report covering all Section A topics for this batch.")

        if st.button("📝 Generate Report", type="primary"):
            html_report = generate_html_report(analysis_df, stats, rules_df, itemsets_df)
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_report.encode('utf-8'),
                file_name=f"carelens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
            st.success("✅ Report ready! Click above to download and open in any browser.")

    else:
        st.info("👆 Upload images or a CSV file above to begin analysis.")


# ============================================================
# PAGE 3: PREPROCESSING VISUALIZER
# ============================================================

elif page == "🔬 Preprocessing Visualizer":
    st.title("🔬 Preprocessing Visualizer")
    st.markdown("See exactly what happens to your X-ray before the model sees it.")
    st.markdown('<span class="section-tag-green">Section A — Unit 3: Data Preprocessing</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    This page gives you a step-by-step visual walkthrough of the exact preprocessing pipeline
    that every X-ray image passes through before being fed into the MediFusion-Net model.

    **Steps visualised:**
    - 🖼️ **Step 1 — Load as RGB:** Converts the image to a standard 3-channel RGB format,
      ensuring consistent input regardless of the original file type.
    - 📐 **Step 2 — Resize to 224×224:** Scales the image to the fixed resolution expected by the model.
    - 🔢 **Step 3 — Array Conversion:** Converts the PIL image into a NumPy array with shape (224, 224, 3).
    - ➗ **Step 4 — Normalisation (÷255):** Scales pixel values from the 0–255 integer range down to
      0.0–1.0 floats. This stabilises training and is standard practice for deep learning.
    - 📦 **Step 5 — Batch Expansion:** Adds a batch dimension, producing shape (1, 224, 224, 3) as
      required by TensorFlow model input.

    **Also shown:** Per-channel (R, G, B) pixel statistics before and after normalisation, and
    a pixel intensity histogram so you can see the brightness distribution of your scan.

    **Why this matters (Unit 3 — Data Preprocessing):** Raw medical images cannot be directly
    fed into a neural network. This preprocessing pipeline handles format standardisation,
    resolution normalisation, and value scaling — equivalent to data cleaning and transformation
    in a classical data mining workflow.
    """)

    uploaded = st.file_uploader(
        "Upload a single X-ray image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="preprocess_uploader"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        st.subheader("🖼️ Preprocessing Pipeline")
        fig = show_preprocessing_pipeline(img)
        st.pyplot(fig)
        st.download_button("💾 Download Pipeline Image", fig_to_bytes(fig),
                           file_name="preprocessing_pipeline.png", mime="image/png")

        st.markdown("---")
        st.subheader("📊 Pixel Statistics (Before vs After Normalization)")

        orig_array = np.array(img.resize((224, 224)))
        norm_array = orig_array / 255.0

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Normalization (raw pixels)**")
            stats_before = pd.DataFrame({
                'Channel': ['Red', 'Green', 'Blue'],
                'Mean': [orig_array[:,:,i].mean() for i in range(3)],
                'Std':  [orig_array[:,:,i].std()  for i in range(3)],
                'Min':  [orig_array[:,:,i].min()  for i in range(3)],
                'Max':  [orig_array[:,:,i].max()  for i in range(3)],
            }).round(3)
            st.dataframe(stats_before, use_container_width=True)

        with col2:
            st.markdown("**After Normalization (÷255)**")
            stats_after = pd.DataFrame({
                'Channel': ['Red', 'Green', 'Blue'],
                'Mean': [norm_array[:,:,i].mean() for i in range(3)],
                'Std':  [norm_array[:,:,i].std()  for i in range(3)],
                'Min':  [norm_array[:,:,i].min()  for i in range(3)],
                'Max':  [norm_array[:,:,i].max()  for i in range(3)],
            }).round(4)
            st.dataframe(stats_after, use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Pixel Intensity Histogram")
        fig_hist, ax = plt.subplots(figsize=(10, 4))
        colors = ['red', 'green', 'blue']
        for i, (c, label) in enumerate(zip(colors, ['Red', 'Green', 'Blue'])):
            ax.hist(orig_array[:,:,i].flatten(), bins=50, color=c, alpha=0.5, label=f'{label} channel')
        ax.set_title("Pixel Intensity Distribution (Original)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Pixel Value (0–255)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_hist)

        st.markdown("---")
        st.subheader("📋 Preprocessing Steps Summary")
        steps_data = {
            'Step': ['1. Load as RGB', '2. Resize', '3. Array Conversion', '4. Normalization', '5. Batch Expansion'],
            'Operation': [
                'img.convert("RGB")',
                'img.resize((224, 224))',
                'np.array(img)',
                'array / 255.0',
                'np.expand_dims(array, axis=0)'
            ],
            'Input Shape': [f'{img.size[0]}×{img.size[1]}', f'{img.size[0]}×{img.size[1]}×3', '224×224×3', '224×224×3', '224×224×3'],
            'Output Shape': [f'{img.size[0]}×{img.size[1]}×3', '224×224×3', '224×224×3', '224×224×3 (float)', '1×224×224×3'],
            'Value Range': ['0–255 (int)', '0–255 (int)', '0–255 (int)', '0.0–1.0 (float)', '0.0–1.0 (float)']
        }
        st.dataframe(pd.DataFrame(steps_data), use_container_width=True)
    else:
        st.info("👆 Upload an X-ray image above to see the preprocessing pipeline.")

# ============================================================
# PAGE 4: SIMILARITY & DISTANCE MATRIX
# ============================================================

elif page == "🧮 Similarity & Distance Matrix":
    st.title("🧮 Image Similarity & Distance Matrix")
    st.markdown('<span class="section-tag">Section A — Unit 2: Measuring Data Similarity and Dissimilarity</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Description box
    st.info("""
    **What this does:**

    This tool compares every X-ray image in your uploaded batch against every other image using
    pixel histogram features. It computes a **pairwise distance matrix** — a table where each cell
    tells you how different two images are from each other (0 = identical, higher = more different).

    - 🔵 **Dark cells** in the heatmap = similar images (small distance)
    - 🔴 **Bright cells** = dissimilar images (large distance)
    - Images that are far from all others are flagged as **potential outliers**

    **How it works:** Each image is converted to a 64-bin grayscale histogram (a compact
    numerical fingerprint). Euclidean distance is then computed between every pair of fingerprints.
    This is a classical data similarity measure from Unit 2 of data mining.

    **Use case:** Spot duplicate or near-duplicate X-rays in a batch, identify unusual scans
    that look very different from the rest, and understand the diversity of your image set.
    """)

    uploaded = st.file_uploader(
        "Upload X-ray images (minimum 2, recommended 5+)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="sim_uploader"
    )
    zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="sim_zip")

    images_list = []
    if uploaded:
        for f in uploaded:
            try:
                img = Image.open(f).convert("L")  # grayscale
                images_list.append((f.name, img))
            except:
                pass
    elif zip_up:
        raw = extract_images_from_zip(zip_up)
        images_list = [(name, img.convert("L")) for name, img in raw]

    if len(images_list) < 2:
        st.warning("Please upload at least 2 images to compute a distance matrix.")
    elif images_list:
        if st.button("🔍 Compute Similarity Matrix", type="primary"):

            # Build histogram fingerprints
            def get_histogram(img: Image.Image, bins=64) -> np.ndarray:
                arr = np.array(img.resize((224, 224))).flatten()
                hist, _ = np.histogram(arr, bins=bins, range=(0, 255))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-8)
                return hist

            names = [n for n, _ in images_list]
            histograms = [get_histogram(img) for _, img in images_list]

            # Euclidean distance matrix
            n = len(histograms)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.sqrt(np.sum((histograms[i] - histograms[j])**2))

            dist_df = pd.DataFrame(dist_matrix, index=names, columns=names).round(4)

            st.subheader("📊 Pairwise Distance Matrix")
            st.dataframe(dist_df, use_container_width=True)
            st.caption("Values represent Euclidean distance between image histogram fingerprints. 0 = identical.")

            # Heatmap
            st.subheader("🗺️ Distance Heatmap")
            short_names = [n[:18] + "…" if len(n) > 18 else n for n in names]
            fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
            sns.heatmap(
                dist_matrix, annot=(n <= 12), fmt='.3f',
                xticklabels=short_names, yticklabels=short_names,
                cmap='YlOrRd', ax=ax, linewidths=0.4,
                cbar_kws={"label": "Euclidean Distance"}
            )
            ax.set_title("Image Similarity / Dissimilarity Matrix", fontsize=13, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("💾 Download Heatmap", fig_to_bytes(fig),
                               file_name="similarity_matrix.png", mime="image/png")

            # Most similar / most dissimilar pairs
            st.subheader("🔗 Most Similar Pairs")
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append({'Image A': names[i], 'Image B': names[j], 'Distance': round(dist_matrix[i,j], 4)})
            pairs_df = pd.DataFrame(pairs).sort_values('Distance')

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔵 Most Similar (smallest distance)**")
                st.dataframe(pairs_df.head(5), use_container_width=True)
            with col2:
                st.markdown("**🔴 Most Dissimilar (largest distance)**")
                st.dataframe(pairs_df.tail(5).sort_values('Distance', ascending=False), use_container_width=True)

            # Outlier detection via average distance
            st.subheader("🚨 Potential Outlier Images")
            avg_distances = dist_matrix.sum(axis=1) / (n - 1)
            threshold = np.mean(avg_distances) + 1.5 * np.std(avg_distances)
            outlier_indices = np.where(avg_distances > threshold)[0]

            outlier_df = pd.DataFrame({
                'Image': names,
                'Avg Distance to Others': avg_distances.round(4),
                'Status': ['⚠️ Outlier' if i in outlier_indices else '✅ Normal' for i in range(n)]
            }).sort_values('Avg Distance to Others', ascending=False)
            st.dataframe(outlier_df, use_container_width=True)

            if len(outlier_indices) > 0:
                st.warning(f"⚠️ {len(outlier_indices)} image(s) flagged as outliers — they look significantly different from the rest of the batch.")
            else:
                st.success("✅ No strong outliers detected — the batch appears visually consistent.")

            csv = dist_df.to_csv()
            st.download_button("💾 Download Distance Matrix as CSV", csv,
                               file_name="distance_matrix.csv", mime="text/csv")


# ============================================================
# PAGE 5: OUTLIER & ANOMALY DETECTION
# ============================================================

elif page == "⚠️ Outlier & Anomaly Detection":
    st.title("⚠️ Outlier & Anomaly Detection")
    st.markdown('<span class="section-tag">Section A — Unit 2: Measures of Dispersion, IQR-based Outlier Detection</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    After your batch is predicted by the classifier, this tool analyses the **confidence scores
    and uncertainty values** to automatically flag images that behave unusually compared to
    the rest of the batch.

    **Three types of anomalies are detected:**
    - 🔴 **Low Confidence Outliers** — images where the model was not sure what class it is.
      These cases need human review before any clinical use.
    - 🟠 **High Epistemic Uncertainty** — the model itself is uncertain (not enough training data
      for this type of image). Could indicate an unusual X-ray pattern.
    - 🟡 **High Aleatoric Uncertainty** — the image itself is inherently ambiguous (noise,
      poor quality, unclear anatomy). The uncertainty is in the data, not the model.

    **Method used:** IQR-based outlier detection — the same statistical method taught in Unit 2.
    Any value beyond Q3 + 1.5×IQR is flagged as an outlier.

    **Use case:** In a clinical screening batch, you want to quickly isolate which scans
    the AI is least sure about so a radiologist can review those first.
    """)

    # Input
    input_method = st.radio(
        "Data source:",
        ["Upload Images (predict + detect)", "Upload CSV (from Classifier)"],
        horizontal=True,
        key="outlier_input"
    )

    outlier_df = None

    if input_method == "Upload Images (predict + detect)":
        uploaded = st.file_uploader(
            "Upload X-ray images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="outlier_img_uploader"
        )
        zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="outlier_zip")

        images_list = []
        if uploaded:
            for f in uploaded:
                try:
                    img = Image.open(f).convert("RGB")
                    images_list.append((f.name, img))
                except:
                    pass
        elif zip_up:
            images_list = extract_images_from_zip(zip_up)

        if images_list:
            if st.button("🔍 Predict & Detect Outliers", type="primary"):
                progress_bar = st.progress(0)
                results, filenames = [], []
                for i, (fname, img) in enumerate(images_list):
                    try:
                        result = predict_single_image(img)
                        results.append(result)
                        filenames.append(fname)
                    except Exception as e:
                        st.warning(f"Skipped {fname}: {e}")
                    progress_bar.progress((i + 1) / len(images_list))
                outlier_df = create_batch_summary(results, filenames)
                st.session_state['outlier_df'] = outlier_df

        if outlier_df is None and 'outlier_df' in st.session_state:
            outlier_df = st.session_state['outlier_df']

    else:
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="outlier_csv")
        if csv_file:
            outlier_df = pd.read_csv(csv_file)
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                if col in outlier_df.columns:
                    outlier_df[col] = pd.to_numeric(outlier_df[col], errors='coerce')
        if outlier_df is None and 'last_batch_df' in st.session_state:
            st.info("💡 Using results from your last Classifier run.")
            outlier_df = st.session_state['last_batch_df'].copy()
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                outlier_df[col] = pd.to_numeric(outlier_df[col], errors='coerce')

    if outlier_df is not None and len(outlier_df) >= 3:

        st.markdown("---")
        st.subheader("📋 Full Batch")
        st.dataframe(outlier_df, use_container_width=True)

        def iqr_outliers(series: pd.Series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return lower, upper, series < lower, series > upper

        conf_vals  = outlier_df['Confidence'].astype(float)
        epis_vals  = outlier_df['Epistemic Uncertainty'].astype(float)
        alea_vals  = outlier_df['Aleatoric Uncertainty'].astype(float)

        _, conf_upper, conf_low_mask, _     = iqr_outliers(conf_vals)
        _, epis_upper, _, epis_high_mask    = iqr_outliers(epis_vals)
        _, alea_upper, _, alea_high_mask    = iqr_outliers(alea_vals)

        # Summary metrics
        st.subheader("📊 Anomaly Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Images", len(outlier_df))
        c2.metric("🔴 Low Confidence", int(conf_low_mask.sum()))
        c3.metric("🟠 High Epistemic Unc.", int(epis_high_mask.sum()))
        c4.metric("🟡 High Aleatoric Unc.", int(alea_high_mask.sum()))

        # Tag each row
        tagged = outlier_df.copy()
        tags = []
        for i in range(len(tagged)):
            t = []
            if conf_low_mask.iloc[i]:   t.append("🔴 Low Confidence")
            if epis_high_mask.iloc[i]:  t.append("🟠 High Epistemic")
            if alea_high_mask.iloc[i]:  t.append("🟡 High Aleatoric")
            tags.append(", ".join(t) if t else "✅ Normal")
        tagged['Anomaly Flag'] = tags
        tagged['Needs Review'] = tagged['Anomaly Flag'].apply(lambda x: "⚠️ Yes" if x != "✅ Normal" else "No")

        # Flagged images
        flagged = tagged[tagged['Needs Review'] == "⚠️ Yes"]
        normal  = tagged[tagged['Needs Review'] == "No"]

        st.subheader("⚠️ Flagged Images — Review Required")
        if len(flagged) > 0:
            st.error(f"{len(flagged)} image(s) flagged for review out of {len(tagged)} total.")
            st.dataframe(flagged, use_container_width=True)
        else:
            st.success("✅ No anomalies detected. All images are within normal ranges.")

        st.subheader("✅ Normal Images")
        st.dataframe(normal[['Filename','Predicted Class','Confidence','Epistemic Uncertainty','Aleatoric Uncertainty']], use_container_width=True)

        # Visualisation
        st.subheader("📈 Outlier Visualisation")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("IQR-Based Outlier Detection", fontsize=13, fontweight='bold')

        datasets = [
            (conf_vals, conf_low_mask, "Confidence", "🔴 Low Confidence Outliers", '#0066cc'),
            (epis_vals, epis_high_mask, "Epistemic Uncertainty", "🟠 High Epistemic Outliers", '#e85d04'),
            (alea_vals, alea_high_mask, "Aleatoric Uncertainty", "🟡 High Aleatoric Outliers", '#d4ac0d'),
        ]

        for ax, (vals, mask, label, title, color) in zip(axes, datasets):
            x = range(len(vals))
            ax.scatter([i for i, m in enumerate(mask) if not m],
                       [v for v, m in zip(vals, mask) if not m],
                       color=color, alpha=0.7, label='Normal', s=60)
            ax.scatter([i for i, m in enumerate(mask) if m],
                       [v for v, m in zip(vals, mask) if m],
                       color='red', alpha=0.9, label='Outlier', s=80, marker='X', zorder=5)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            ax.axhline(q3 + 1.5 * iqr, color='red', linestyle='--', linewidth=1, label=f'Q3+1.5×IQR')
            ax.axhline(q1 - 1.5 * iqr, color='orange', linestyle='--', linewidth=1, label=f'Q1-1.5×IQR')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Image Index")
            ax.set_ylabel(label)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("💾 Download Outlier Plot", fig_to_bytes(fig),
                           file_name="outlier_detection.png", mime="image/png")

        # IQR stats table
        st.subheader("📐 IQR Statistics Used for Detection")
        iqr_stats = []
        for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
            vals = outlier_df[col].astype(float)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            iqr_stats.append({
                'Metric': col,
                'Q1': round(q1, 4),
                'Q3': round(q3, 4),
                'IQR': round(iqr, 4),
                'Lower Fence (Q1-1.5×IQR)': round(q1 - 1.5*iqr, 4),
                'Upper Fence (Q3+1.5×IQR)': round(q3 + 1.5*iqr, 4),
            })
        st.dataframe(pd.DataFrame(iqr_stats), use_container_width=True)

        csv = tagged.to_csv(index=False)
        st.download_button("💾 Download Flagged Results as CSV", csv,
                           file_name="outlier_results.csv", mime="text/csv")

    elif outlier_df is not None:
        st.warning("Upload at least 3 images to run outlier detection meaningfully.")
    else:
        st.info("👆 Upload images or a CSV file above to begin.")


# ============================================================
# PAGE 6: DATA QUALITY CHECKER
# ============================================================

elif page == "🛡️ Data Quality Checker":
    st.title("🛡️ Data Quality Checker")
    st.markdown('<span class="section-tag-green">Section A — Unit 3: Data Preprocessing — Dealing with Noisy / Low Quality Data</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    Before sending images to the AI model, this tool automatically inspects each uploaded
    X-ray for common **data quality problems** that could affect prediction accuracy.

    **Checks performed on each image:**
    - 🌑 **Too Dark** — average pixel brightness below threshold. May be underexposed or corrupted.
    - ☀️ **Too Bright / Overexposed** — average brightness too high. Washed-out images lose detail.
    - 🌫️ **Too Blurry** — measured using Laplacian variance. Low sharpness = poor image quality.
    - 📐 **Wrong Aspect Ratio** — extremely wide or tall images are unlikely to be chest X-rays.
    - 📏 **Low Resolution** — images smaller than 100×100 pixels may not have enough detail.
    - 🎨 **Not Grayscale** — chest X-rays are typically grayscale. Colour images may not be X-rays.

    **Why this matters (Unit 3 — Data Preprocessing):** In data mining, handling noisy and
    low-quality data before analysis is a critical preprocessing step. Feeding bad-quality
    images to the model is equivalent to noisy data in a dataset — it degrades results.

    **Output:** Each image gets a quality score and pass/fail flag. Failed images are listed
    separately so you can remove or replace them before running the classifier.
    """)

    uploaded = st.file_uploader(
        "Upload X-ray images to check quality",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="quality_uploader"
    )
    zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="quality_zip")

    images_list = []
    if uploaded:
        for f in uploaded:
            try:
                img = Image.open(f).convert("RGB")
                images_list.append((f.name, img))
            except:
                pass
    elif zip_up:
        images_list = extract_images_from_zip(zip_up)

    # Thresholds — shown to user and adjustable
    st.subheader("⚙️ Quality Thresholds")
    st.markdown("You can adjust these thresholds based on your dataset.")
    col1, col2, col3 = st.columns(3)
    with col1:
        dark_thresh  = st.slider("Min brightness (too dark below)", 0, 100, 20)
        bright_thresh = st.slider("Max brightness (too bright above)", 150, 255, 220)
    with col2:
        blur_thresh  = st.slider("Min sharpness / Laplacian variance (too blurry below)", 0, 200, 30)
        min_res      = st.slider("Min resolution (px, shorter side)", 50, 300, 100)
    with col3:
        min_ar = st.slider("Min aspect ratio (W/H)", 0.3, 1.0, 0.5)
        max_ar = st.slider("Max aspect ratio (W/H)", 1.0, 3.0, 2.0)

    def check_image_quality(filename: str, img: Image.Image) -> Dict[str, Any]:
        arr_rgb  = np.array(img.resize((224, 224)))
        arr_gray = np.array(img.convert('L').resize((224, 224)))
        w, h     = img.size

        brightness  = float(arr_gray.mean())
        sharpness   = float(cv2.Laplacian(arr_gray, cv2.CV_64F).var())
        aspect_ratio = round(w / h, 3)
        min_dim     = min(w, h)

        # Check if likely grayscale (R≈G≈B)
        r_mean = arr_rgb[:,:,0].mean()
        g_mean = arr_rgb[:,:,1].mean()
        b_mean = arr_rgb[:,:,2].mean()
        channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
        is_color = channel_diff > 15

        issues = []
        if brightness < dark_thresh:    issues.append(f"🌑 Too dark (brightness={brightness:.1f})")
        if brightness > bright_thresh:  issues.append(f"☀️ Too bright (brightness={brightness:.1f})")
        if sharpness < blur_thresh:     issues.append(f"🌫️ Too blurry (sharpness={sharpness:.1f})")
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            issues.append(f"📐 Unusual aspect ratio ({aspect_ratio})")
        if min_dim < min_res:           issues.append(f"📏 Low resolution ({w}×{h})")
        if is_color:                    issues.append(f"🎨 Appears to be colour image (may not be X-ray)")

        quality_score = max(0, 100 - len(issues) * 20)

        return {
            'Filename': filename,
            'Width': w,
            'Height': h,
            'Aspect Ratio': aspect_ratio,
            'Brightness': round(brightness, 2),
            'Sharpness': round(sharpness, 2),
            'Is Colour': is_color,
            'Issues': "; ".join(issues) if issues else "None",
            'Issue Count': len(issues),
            'Quality Score': quality_score,
            'Status': "❌ Fail" if issues else "✅ Pass"
        }

    if images_list:
        if st.button("🛡️ Run Quality Check", type="primary"):
            quality_results = []
            progress_bar = st.progress(0)

            for i, (fname, img) in enumerate(images_list):
                result = check_image_quality(fname, img)
                quality_results.append(result)
                progress_bar.progress((i + 1) / len(images_list))

            quality_df = pd.DataFrame(quality_results)
            st.session_state['quality_df'] = quality_df

        if 'quality_df' in st.session_state:
            quality_df = st.session_state['quality_df']

            st.markdown("---")

            # Summary
            passed = (quality_df['Status'] == "✅ Pass").sum()
            failed = (quality_df['Status'] == "❌ Fail").sum()

            st.subheader("📊 Quality Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Images", len(quality_df))
            c2.metric("✅ Passed", int(passed))
            c3.metric("❌ Failed", int(failed))
            c4.metric("Pass Rate", f"{passed/len(quality_df)*100:.1f}%")

            # Full table
            st.subheader("📋 Full Quality Report")
            st.dataframe(quality_df, use_container_width=True)

            # Failed images
            failed_df = quality_df[quality_df['Status'] == "❌ Fail"]
            if len(failed_df) > 0:
                st.subheader("❌ Images with Quality Issues")
                st.error(f"{len(failed_df)} image(s) failed quality checks. Consider removing or replacing them before classification.")
                for _, row in failed_df.iterrows():
                    with st.expander(f"❌ {row['Filename']} — Score: {row['Quality Score']}/100"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Resolution:** {row['Width']}×{row['Height']}")
                            st.write(f"**Aspect Ratio:** {row['Aspect Ratio']}")
                            st.write(f"**Brightness:** {row['Brightness']}")
                            st.write(f"**Sharpness:** {row['Sharpness']}")
                        with col2:
                            st.write(f"**Issues Found:**")
                            for issue in row['Issues'].split("; "):
                                st.write(f"  - {issue}")
            else:
                st.success("✅ All images passed quality checks!")

            # Visual — quality score bar
            st.subheader("📈 Quality Scores per Image")
            fig, ax = plt.subplots(figsize=(max(8, len(quality_df) * 0.7), 5))
            colors = ['#28a745' if s == "✅ Pass" else '#dc3545' for s in quality_df['Status']]
            short_names = [n[:15] + "…" if len(n) > 15 else n for n in quality_df['Filename']]
            bars = ax.bar(short_names, quality_df['Quality Score'], color=colors, edgecolor='white', alpha=0.85)
            ax.axhline(60, color='orange', linestyle='--', linewidth=1.5, label='Min acceptable (60)')
            ax.set_ylim(0, 110)
            ax.set_title("Image Quality Scores", fontsize=13, fontweight='bold')
            ax.set_xlabel("Image")
            ax.set_ylabel("Quality Score (0–100)")
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            for bar, score in zip(bars, quality_df['Quality Score']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(score), ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("💾 Download Quality Chart", fig_to_bytes(fig),
                               file_name="quality_scores.png", mime="image/png")

            # Issue breakdown
            st.subheader("🔍 Issue Breakdown Across Batch")
            all_issues = []
            for issues_str in quality_df[quality_df['Issues'] != 'None']['Issues']:
                for issue in issues_str.split("; "):
                    # Extract just the label
                    label = issue.split("(")[0].strip()
                    all_issues.append(label)

            if all_issues:
                issue_counts = pd.Series(all_issues).value_counts()
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.barh(issue_counts.index, issue_counts.values, color='#e85d04', alpha=0.8, edgecolor='white')
                ax2.set_title("Frequency of Quality Issues Across Batch", fontsize=12, fontweight='bold')
                ax2.set_xlabel("Number of Images Affected")
                ax2.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.success("No issues found across the batch.")

            csv = quality_df.to_csv(index=False)
            st.download_button("💾 Download Quality Report as CSV", csv,
                               file_name="quality_report.csv", mime="text/csv")
    else:
        st.info("👆 Upload images above to run the quality checker.")



# ============================================================
# PAGE 7: k-NN SIMILARITY FINDER
# ============================================================

elif page == "🔎 k-NN Similarity Finder":
    st.title("🔎 k-NN Similarity Finder")
    st.markdown('<span class="section-tag">Section B — Unit 6: k-Nearest Neighbour Classification</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    This tool implements **k-Nearest Neighbour (k-NN)** similarity search on chest X-ray images.
    Upload a single **query image** and a **batch of reference images** — the tool will extract
    a compact feature fingerprint (64-bin grayscale histogram) from each image and use
    **Euclidean distance** to find the k most visually similar images in the batch.

    **How it works:**
    - 📷 Each image is converted to grayscale and resized to 224×224.
    - 📊 A 64-bin normalised intensity histogram is computed — a compact numeric descriptor of brightness distribution.
    - 📏 Euclidean distance is calculated between the query image's histogram and every reference image's histogram.
    - 🏆 The k reference images with the **smallest distance** are returned as nearest neighbours.

    **Why this matters (Unit 6 — k-NN):** k-NN is a non-parametric, instance-based classifier.
    Rather than training a model, it classifies a new point by the majority label of its k closest
    neighbours in feature space. Here we use the same distance metric for retrieval — finding
    the most similar scans to a query, which can help radiologists cross-reference similar historical cases.

    **Output:** A ranked list and side-by-side display of the top-k most similar images, with
    their distance scores. Lower distance = more similar.
    """)

    col_q, col_r = st.columns(2)
    with col_q:
        query_file = st.file_uploader(
            "📷 Upload Query Image (the image to match)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="knn_query"
        )
    with col_r:
        ref_files = st.file_uploader(
            "📁 Upload Reference Batch (images to search through)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="knn_refs"
        )
    ref_zip = st.file_uploader("Or upload a ZIP for the reference batch", type=["zip"], key="knn_zip")

    k = st.slider("k — Number of nearest neighbours to return", min_value=1, max_value=10, value=3)

    ref_images = []
    if ref_files:
        for f in ref_files:
            try:
                img = Image.open(f).convert("RGB")
                ref_images.append((f.name, img))
            except:
                pass
    elif ref_zip:
        ref_images = extract_images_from_zip(ref_zip)

    if query_file and len(ref_images) >= 1:
        if st.button("🔍 Find Nearest Neighbours", type="primary"):

            query_img = Image.open(query_file).convert("RGB")

            def histogram_fingerprint(img: Image.Image, bins=64) -> np.ndarray:
                arr = np.array(img.convert("L").resize((224, 224))).flatten()
                hist, _ = np.histogram(arr, bins=bins, range=(0, 255))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-8)
                return hist

            query_hist = histogram_fingerprint(query_img)
            distances = []
            for name, img in ref_images:
                h = histogram_fingerprint(img)
                dist = float(np.sqrt(np.sum((query_hist - h) ** 2)))
                distances.append((name, img, dist))

            distances.sort(key=lambda x: x[2])
            top_k = distances[:k]

            st.markdown("---")
            st.subheader(f"🖼️ Query Image")
            st.image(query_img, caption=f"Query: {query_file.name}", width=250)

            st.subheader(f"🏆 Top-{k} Nearest Neighbours")
            rank_cols = st.columns(min(k, 5))
            for rank, (name, img, dist) in enumerate(top_k):
                col_idx = rank % min(k, 5)
                with rank_cols[col_idx]:
                    st.image(img, caption=f"#{rank+1} — {name}", use_column_width=True)
                    st.markdown(f"**Distance:** `{dist:.4f}`")
                    if dist < 0.05:
                        st.success("Very similar")
                    elif dist < 0.15:
                        st.info("Similar")
                    else:
                        st.warning("Dissimilar")

            st.markdown("---")
            st.subheader("📊 Distance Ranking Table")
            ranking_df = pd.DataFrame([
                {"Rank": i+1, "Filename": name, "Euclidean Distance": round(dist, 4),
                 "Similarity": "Very High" if dist < 0.05 else ("High" if dist < 0.15 else "Low")}
                for i, (name, _, dist) in enumerate(distances)
            ])
            st.dataframe(ranking_df, use_container_width=True)
            st.caption("Lower Euclidean distance = more similar image. Distance of 0 = identical histogram fingerprint.")

            fig, ax = plt.subplots(figsize=(max(8, len(distances) * 0.6), 4))
            colors = ['#0066cc' if i < k else '#cbd5e1' for i in range(len(distances))]
            short_names = [n[:14] + "…" if len(n) > 14 else n for n, _, _ in distances]
            bars = ax.bar(short_names, [d for _, _, d in distances], color=colors, edgecolor='white')
            ax.axvline(k - 0.5, color='red', linestyle='--', linewidth=1.5, label=f'k={k} cutoff')
            ax.set_title("Euclidean Distance from Query Image (sorted)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Reference Image")
            ax.set_ylabel("Distance")
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("💾 Download Distance Chart", fig_to_bytes(fig),
                               file_name="knn_distances.png", mime="image/png")

            csv = ranking_df.to_csv(index=False)
            st.download_button("💾 Download Ranking as CSV", csv,
                               file_name="knn_ranking.csv", mime="text/csv")

    elif query_file and len(ref_images) == 0:
        st.warning("Please upload at least 1 reference image or a ZIP batch to search through.")
    else:
        st.info("👆 Upload a query image and a reference batch above, then click Find Nearest Neighbours.")


# ============================================================
# PAGE 8: UNCERTAINTY ANALYSIS DASHBOARD
# ============================================================

elif page == "📉 Uncertainty Analysis Dashboard":
    st.title("📉 Uncertainty Analysis Dashboard")
    st.markdown('<span class="section-tag">Section A — Unit 5: Bayesian Uncertainty — Epistemic vs Aleatoric</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    This dashboard performs a deep-dive analysis of the **two types of uncertainty** produced
    by MediFusion-Net's Bayesian prediction head, going beyond simply flagging outliers.

    **The two uncertainty types explained:**
    - 🔍 **Epistemic Uncertainty** (model uncertainty) — the model doesn't know enough.
      Caused by lack of training data for this type of image. *Can be reduced by collecting more data.*
      High epistemic = "I haven't seen an X-ray like this before."
    - ⚡ **Aleatoric Uncertainty** (data uncertainty) — the image itself is inherently ambiguous.
      Caused by image noise, poor scan quality, or genuinely overlapping visual features between classes.
      *Cannot be reduced by more data.* High aleatoric = "Even a perfect model would struggle here."

    **4-Quadrant Classification:**
    Each prediction is placed into one of four quadrants based on whether its epistemic and
    aleatoric uncertainties are above or below the batch median:
    - ✅ **Q1 — Confident & Clear:** Low epistemic, low aleatoric. Model is sure AND image is unambiguous. Most reliable predictions.
    - 🟡 **Q2 — Model Uncertain, Image Clear:** High epistemic, low aleatoric. Good image quality but model hasn't seen this pattern. Collect more similar training data.
    - 🟠 **Q3 — Model OK, Image Ambiguous:** Low epistemic, high aleatoric. Model is trained well but the scan is noisy/unclear. Improve image acquisition.
    - 🔴 **Q4 — Doubly Uncertain:** High epistemic, high aleatoric. Both the model and the image are uncertain. Requires radiologist review.

    **Why this matters (Unit 5):** Understanding the *source* of uncertainty is critical in medical AI.
    Treating epistemic and aleatoric uncertainty the same leads to wrong conclusions about how to improve the system.
    """)

    input_method = st.radio(
        "Data source:",
        ["Upload Images (predict + analyse)", "Upload CSV (from Classifier)"],
        horizontal=True,
        key="unc_input"
    )

    unc_df = None

    if input_method == "Upload Images (predict + analyse)":
        uploaded = st.file_uploader(
            "Upload X-ray images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="unc_img_uploader"
        )
        zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="unc_zip")

        images_list = []
        if uploaded:
            for f in uploaded:
                try:
                    img = Image.open(f).convert("RGB")
                    images_list.append((f.name, img))
                except:
                    pass
        elif zip_up:
            images_list = extract_images_from_zip(zip_up)

        if images_list:
            if st.button("🔍 Predict & Analyse Uncertainty", type="primary"):
                progress_bar = st.progress(0)
                results, filenames = [], []
                for i, (fname, img) in enumerate(images_list):
                    try:
                        result = predict_single_image(img)
                        results.append(result)
                        filenames.append(fname)
                    except Exception as e:
                        st.warning(f"Skipped {fname}: {e}")
                    progress_bar.progress((i + 1) / len(images_list))
                unc_df = create_batch_summary(results, filenames)
                st.session_state['unc_df'] = unc_df
                st.success(f"✅ Predicted {len(results)} images.")

        if unc_df is None and 'unc_df' in st.session_state:
            unc_df = st.session_state['unc_df']

    else:
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="unc_csv")
        if csv_file:
            unc_df = pd.read_csv(csv_file)
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                if col in unc_df.columns:
                    unc_df[col] = pd.to_numeric(unc_df[col], errors='coerce')
            st.success(f"✅ Loaded {len(unc_df)} records.")
        if unc_df is None and 'last_batch_df' in st.session_state:
            st.info("💡 Using results from your last Classifier run.")
            unc_df = st.session_state['last_batch_df'].copy()
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                unc_df[col] = pd.to_numeric(unc_df[col], errors='coerce')

    if unc_df is not None and len(unc_df) >= 2:
        epis = unc_df['Epistemic Uncertainty'].astype(float)
        alea = unc_df['Aleatoric Uncertainty'].astype(float)
        conf = unc_df['Confidence'].astype(float)

        epis_med = epis.median()
        alea_med = alea.median()

        def assign_quadrant(e, a):
            if e <= epis_med and a <= alea_med:
                return "Q1 ✅ Confident & Clear"
            elif e > epis_med and a <= alea_med:
                return "Q2 🟡 Model Uncertain"
            elif e <= epis_med and a > alea_med:
                return "Q3 🟠 Image Ambiguous"
            else:
                return "Q4 🔴 Doubly Uncertain"

        unc_df = unc_df.copy()
        unc_df['Quadrant'] = [assign_quadrant(e, a) for e, a in zip(epis, alea)]

        st.markdown("---")

        # Summary metrics
        st.subheader("📊 Uncertainty Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Epistemic", f"{epis.mean():.4f}", help="Average model uncertainty across batch")
        c2.metric("Mean Aleatoric", f"{alea.mean():.4f}", help="Average data uncertainty across batch")
        c3.metric("Mean Confidence", f"{conf.mean():.4f}")
        c4.metric("Doubly Uncertain (Q4)", int((unc_df['Quadrant'] == "Q4 🔴 Doubly Uncertain").sum()))

        # Quadrant summary
        st.subheader("🔲 4-Quadrant Classification")
        quad_counts = unc_df['Quadrant'].value_counts()
        qc1, qc2, qc3, qc4 = st.columns(4)
        for col_widget, label, desc in zip(
            [qc1, qc2, qc3, qc4],
            ["Q1 ✅ Confident & Clear", "Q2 🟡 Model Uncertain", "Q3 🟠 Image Ambiguous", "Q4 🔴 Doubly Uncertain"],
            ["Most reliable predictions", "Collect more training data", "Improve image quality", "Needs radiologist review"]
        ):
            count = int(quad_counts.get(label, 0))
            col_widget.metric(label, count)
            col_widget.caption(desc)

        # Scatter plot — the main quadrant visualisation
        st.subheader("🗺️ Epistemic vs Aleatoric Uncertainty — Quadrant Plot")
        quad_colors = {
            "Q1 ✅ Confident & Clear":   "#10b981",
            "Q2 🟡 Model Uncertain":      "#f59e0b",
            "Q3 🟠 Image Ambiguous":      "#f97316",
            "Q4 🔴 Doubly Uncertain":     "#ef4444",
        }
        fig, ax = plt.subplots(figsize=(9, 7))
        for quad, grp in unc_df.groupby('Quadrant'):
            ax.scatter(
                grp['Epistemic Uncertainty'].astype(float),
                grp['Aleatoric Uncertainty'].astype(float),
                label=quad, color=quad_colors.get(quad, '#888'),
                s=90, alpha=0.85, edgecolors='white', linewidths=0.5
            )
            for _, row in grp.iterrows():
                ax.annotate(row['Filename'][:10], 
                            (float(row['Epistemic Uncertainty']), float(row['Aleatoric Uncertainty'])),
                            fontsize=6, alpha=0.6, xytext=(3, 3), textcoords='offset points')

        ax.axvline(epis_med, color='#94a3b8', linestyle='--', linewidth=1.2, label=f'Epistemic median ({epis_med:.3f})')
        ax.axhline(alea_med, color='#64748b', linestyle='--', linewidth=1.2, label=f'Aleatoric median ({alea_med:.3f})')

        # Quadrant labels
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.text(x_min + (epis_med - x_min)*0.1, alea_med + (y_max - alea_med)*0.85, "Q1\nConfident\n& Clear",
                fontsize=8, color='#10b981', fontweight='bold', alpha=0.7)
        ax.text(epis_med + (x_max - epis_med)*0.1, alea_med + (y_max - alea_med)*0.85, "Q2\nModel\nUncertain",
                fontsize=8, color='#f59e0b', fontweight='bold', alpha=0.7)
        ax.text(x_min + (epis_med - x_min)*0.1, y_min + (alea_med - y_min)*0.05, "Q3\nImage\nAmbiguous",
                fontsize=8, color='#f97316', fontweight='bold', alpha=0.7)
        ax.text(epis_med + (x_max - epis_med)*0.1, y_min + (alea_med - y_min)*0.05, "Q4\nDoubly\nUncertain",
                fontsize=8, color='#ef4444', fontweight='bold', alpha=0.7)

        ax.set_xlabel("Epistemic Uncertainty (Model Uncertainty)", fontsize=11)
        ax.set_ylabel("Aleatoric Uncertainty (Data Uncertainty)", fontsize=11)
        ax.set_title("Uncertainty Quadrant Plot", fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("💾 Download Quadrant Plot", fig_to_bytes(fig),
                           file_name="uncertainty_quadrants.png", mime="image/png")

        # Uncertainty distribution plots
        st.subheader("📈 Uncertainty Distributions")
        fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(epis, bins=15, color='#e85d04', alpha=0.75, edgecolor='white')
        axes[0].axvline(epis.mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean={epis.mean():.3f}')
        axes[0].axvline(epis_med, color='red', linestyle=':', linewidth=1.5, label=f'Median={epis_med:.3f}')
        axes[0].set_title("Epistemic Uncertainty Distribution", fontweight='bold')
        axes[0].set_xlabel("Epistemic Uncertainty"); axes[0].set_ylabel("Frequency")
        axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

        axes[1].hist(alea, bins=15, color='#d4ac0d', alpha=0.75, edgecolor='white')
        axes[1].axvline(alea.mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean={alea.mean():.3f}')
        axes[1].axvline(alea_med, color='red', linestyle=':', linewidth=1.5, label=f'Median={alea_med:.3f}')
        axes[1].set_title("Aleatoric Uncertainty Distribution", fontweight='bold')
        axes[1].set_xlabel("Aleatoric Uncertainty"); axes[1].set_ylabel("Frequency")
        axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)

        # Per-class breakdown
        st.subheader("🗂️ Uncertainty by Predicted Class")
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        classes = unc_df['Predicted Class'].unique()
        epis_by_class = [unc_df[unc_df['Predicted Class'] == c]['Epistemic Uncertainty'].astype(float).values for c in classes]
        alea_by_class = [unc_df[unc_df['Predicted Class'] == c]['Aleatoric Uncertainty'].astype(float).values for c in classes]

        bp1 = axes3[0].boxplot(epis_by_class, labels=classes, patch_artist=True)
        for patch in bp1['boxes']: patch.set_facecolor('#e85d04'); patch.set_alpha(0.7)
        axes3[0].set_title("Epistemic Uncertainty by Class", fontweight='bold')
        axes3[0].set_ylabel("Epistemic Uncertainty"); axes3[0].grid(axis='y', alpha=0.3)

        bp2 = axes3[1].boxplot(alea_by_class, labels=classes, patch_artist=True)
        for patch in bp2['boxes']: patch.set_facecolor('#d4ac0d'); patch.set_alpha(0.7)
        axes3[1].set_title("Aleatoric Uncertainty by Class", fontweight='bold')
        axes3[1].set_ylabel("Aleatoric Uncertainty"); axes3[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)

        # Full table with quadrant labels
        st.subheader("📋 Full Results with Quadrant Labels")
        st.dataframe(unc_df[['Filename', 'Predicted Class', 'Confidence',
                              'Epistemic Uncertainty', 'Aleatoric Uncertainty', 'Quadrant']],
                     use_container_width=True)

        csv = unc_df.to_csv(index=False)
        st.download_button("💾 Download Uncertainty Report as CSV", csv,
                           file_name="uncertainty_analysis.csv", mime="text/csv")

    elif unc_df is not None:
        st.warning("Upload at least 2 images for a meaningful uncertainty analysis.")
    else:
        st.info("👆 Upload images or a CSV file above to begin.")


# ============================================================
# PAGE 9: BIAS / VARIANCE EXPLORER
# ============================================================

elif page == "📐 Bias / Variance Explorer":
    st.title("📐 Bias / Variance Explorer")
    st.markdown('<span class="section-tag">Section A — Unit 5: Bias–Variance Tradeoff in Model Predictions</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    **What this does:**

    This page analyses how **consistent** the model is across each predicted class by examining
    the **spread (variance) of confidence scores** within each class group.

    **Key concepts:**
    - 📉 **Bias** — systematic error. A class with consistently *low* average confidence suggests
      the model is biased against it — it predicts that class with low certainty across the board.
      This often indicates under-representation in training data.
    - 📊 **Variance** — inconsistency. A class with *high spread* (high standard deviation) in
      confidence means the model is unstable on that class — sometimes very confident, sometimes not.
      High variance = the model hasn't reliably learned the features of that class.

    **The Bias–Variance Tradeoff (Unit 5):** In machine learning, a model suffers from either
    high bias (too simple, underfits the data) or high variance (too sensitive, overfits).
    Examining per-class confidence distributions lets you identify which classes the model
    underfits (high bias) vs which it is inconsistent on (high variance).

    **What to look for:**
    - 🔴 **Low mean confidence + low std** → High bias. Consistent but wrong. More training data for this class needed.
    - 🟠 **Low mean confidence + high std** → High bias AND variance. Poorly learned class.
    - 🟡 **High mean confidence + high std** → High variance. Sometimes correct but unstable.
    - ✅ **High mean confidence + low std** → Low bias, low variance. Well-learned class.

    **Output:** Per-class confidence distributions, variance decomposition chart, and a bias/variance
    classification table for each predicted class.
    """)

    input_method = st.radio(
        "Data source:",
        ["Upload Images (predict + analyse)", "Upload CSV (from Classifier)"],
        horizontal=True,
        key="bv_input"
    )

    bv_df = None

    if input_method == "Upload Images (predict + analyse)":
        uploaded = st.file_uploader(
            "Upload X-ray images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="bv_img_uploader"
        )
        zip_up = st.file_uploader("Or upload a ZIP file", type=["zip"], key="bv_zip")

        images_list = []
        if uploaded:
            for f in uploaded:
                try:
                    img = Image.open(f).convert("RGB")
                    images_list.append((f.name, img))
                except:
                    pass
        elif zip_up:
            images_list = extract_images_from_zip(zip_up)

        if images_list:
            if st.button("🔍 Predict & Explore Bias/Variance", type="primary"):
                progress_bar = st.progress(0)
                results, filenames = [], []
                for i, (fname, img) in enumerate(images_list):
                    try:
                        result = predict_single_image(img)
                        results.append(result)
                        filenames.append(fname)
                    except Exception as e:
                        st.warning(f"Skipped {fname}: {e}")
                    progress_bar.progress((i + 1) / len(images_list))
                bv_df = create_batch_summary(results, filenames)
                st.session_state['bv_df'] = bv_df
                st.success(f"✅ Predicted {len(results)} images.")

        if bv_df is None and 'bv_df' in st.session_state:
            bv_df = st.session_state['bv_df']

    else:
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="bv_csv")
        if csv_file:
            bv_df = pd.read_csv(csv_file)
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                if col in bv_df.columns:
                    bv_df[col] = pd.to_numeric(bv_df[col], errors='coerce')
            st.success(f"✅ Loaded {len(bv_df)} records.")
        if bv_df is None and 'last_batch_df' in st.session_state:
            st.info("💡 Using results from your last Classifier run.")
            bv_df = st.session_state['last_batch_df'].copy()
            for col in ['Confidence', 'Epistemic Uncertainty', 'Aleatoric Uncertainty']:
                bv_df[col] = pd.to_numeric(bv_df[col], errors='coerce')

    if bv_df is not None and len(bv_df) >= 2:
        classes = sorted(bv_df['Predicted Class'].unique())
        conf = bv_df['Confidence'].astype(float)

        # Confidence thresholds
        high_conf_thresh = st.slider("High confidence threshold (mean above = low bias)", 0.5, 1.0, 0.75, 0.01)
        high_var_thresh  = st.slider("High variance threshold (std above = high variance)", 0.0, 0.3, 0.10, 0.01)

        st.markdown("---")

        # Per-class stats
        class_stats = []
        for cls in classes:
            grp = bv_df[bv_df['Predicted Class'] == cls]['Confidence'].astype(float)
            mean_c = grp.mean()
            std_c  = grp.std() if len(grp) > 1 else 0.0
            high_bias = mean_c < high_conf_thresh
            high_var  = std_c > high_var_thresh
            if high_bias and high_var:
                profile = "🔴 High Bias + High Variance"
            elif high_bias:
                profile = "🟠 High Bias (Low Confidence)"
            elif high_var:
                profile = "🟡 High Variance (Unstable)"
            else:
                profile = "✅ Well-Learned (Low Bias, Low Variance)"
            class_stats.append({
                'Class': cls,
                'Count': len(grp),
                'Mean Confidence': round(mean_c, 4),
                'Std Dev (Variance Proxy)': round(std_c, 4),
                'Min Confidence': round(grp.min(), 4),
                'Max Confidence': round(grp.max(), 4),
                'Profile': profile
            })

        stats_df = pd.DataFrame(class_stats)

        # Summary cards
        st.subheader("📋 Per-Class Bias / Variance Profile")
        for _, row in stats_df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 3])
            col1.markdown(f"**{row['Class']}**")
            col2.metric("Mean Conf.", f"{row['Mean Confidence']:.3f}")
            col3.metric("Std Dev", f"{row['Std Dev (Variance Proxy)']:.3f}")
            col4.metric("n", int(row['Count']))
            col5.markdown(row['Profile'])

        st.markdown("---")

        # Confidence distribution per class — violin/box
        st.subheader("📊 Confidence Distributions per Class")
        conf_data = [bv_df[bv_df['Predicted Class'] == c]['Confidence'].astype(float).values for c in classes]

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 2.5), 5))
        colors = ['#0066cc', '#e85d04', '#2d6a4f', '#8b5cf6', '#f43f5e']
        bp = ax.boxplot(conf_data, labels=classes, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.axhline(high_conf_thresh, color='red', linestyle='--', linewidth=1.5,
                   label=f'Bias threshold ({high_conf_thresh})')
        ax.set_title("Confidence Distribution per Predicted Class", fontsize=13, fontweight='bold')
        ax.set_ylabel("Confidence Score")
        ax.set_xlabel("Predicted Class")
        ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("💾 Download Distribution Plot", fig_to_bytes(fig),
                           file_name="bias_variance_distributions.png", mime="image/png")

        # Mean ± std bar chart
        st.subheader("📐 Mean Confidence ± Std Dev per Class")
        fig2, ax2 = plt.subplots(figsize=(max(7, len(classes) * 2), 5))
        means = [row['Mean Confidence'] for _, row in stats_df.iterrows()]
        stds  = [row['Std Dev (Variance Proxy)'] for _, row in stats_df.iterrows()]
        x_pos = range(len(classes))
        bar_colors = ['#10b981' if m >= high_conf_thresh else '#ef4444' for m in means]
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=6, color=bar_colors,
                       alpha=0.8, edgecolor='white', error_kw={'elinewidth': 2, 'ecolor': '#334155'})
        ax2.axhline(high_conf_thresh, color='red', linestyle='--', linewidth=1.5,
                    label=f'Bias threshold ({high_conf_thresh})')
        for bar, std in zip(bars, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.05,
                     f'±{std:.3f}', ha='center', va='bottom', fontsize=9, color='#334155')
        ax2.set_xticks(x_pos); ax2.set_xticklabels(classes)
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Mean Confidence ± Std Dev (Error Bars = Variance)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Predicted Class")
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        st.download_button("💾 Download Mean±Std Chart", fig_to_bytes(fig2),
                           file_name="bias_variance_bars.png", mime="image/png")

        # Variance decomposition scatter
        st.subheader("🎯 Bias–Variance Map (Mean Confidence vs Std Dev)")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        scatter_colors = [('#ef4444' if 'High Bias + High' in r['Profile']
                           else '#f97316' if 'High Bias' in r['Profile']
                           else '#f59e0b' if 'High Variance' in r['Profile']
                           else '#10b981')
                          for _, r in stats_df.iterrows()]
        for i, (_, row) in enumerate(stats_df.iterrows()):
            ax3.scatter(row['Mean Confidence'], row['Std Dev (Variance Proxy)'],
                        color=scatter_colors[i], s=200, zorder=5, edgecolors='white', linewidths=1.5)
            ax3.annotate(row['Class'], (row['Mean Confidence'], row['Std Dev (Variance Proxy)']),
                         xytext=(6, 4), textcoords='offset points', fontsize=10, fontweight='bold')
        ax3.axvline(high_conf_thresh, color='red', linestyle='--', linewidth=1.2, label='Bias threshold')
        ax3.axhline(high_var_thresh,  color='orange', linestyle='--', linewidth=1.2, label='Variance threshold')
        # Quadrant labels
        ax3.text(0.02, high_var_thresh + 0.005, "HIGH VARIANCE →", fontsize=7, color='#f97316', alpha=0.7)
        ax3.text(0.02, 0.005, "LOW VARIANCE →", fontsize=7, color='#10b981', alpha=0.7)
        ax3.set_xlabel("Mean Confidence (lower = higher bias)", fontsize=10)
        ax3.set_ylabel("Std Dev of Confidence (higher = higher variance)", fontsize=10)
        ax3.set_title("Bias–Variance Map per Class", fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1.05); ax3.set_ylim(0, max(stds) * 1.4 + 0.02)
        ax3.legend(fontsize=8); ax3.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig3)
        st.download_button("💾 Download Bias–Variance Map", fig_to_bytes(fig3),
                           file_name="bias_variance_map.png", mime="image/png")

        st.subheader("📋 Full Bias / Variance Table")
        st.dataframe(stats_df, use_container_width=True)

        csv = stats_df.to_csv(index=False)
        st.download_button("💾 Download Table as CSV", csv,
                           file_name="bias_variance_report.csv", mime="text/csv")

    elif bv_df is not None:
        st.warning("Upload at least 2 images to explore bias/variance.")
    else:
        st.info("👆 Upload images or a CSV file above to begin.")


# === Footer ===
st.markdown("---")
st.markdown("🩺 **Carelens AI** — Advanced Medical Image Analysis | MediFusion-Net | Built with Streamlit & TensorFlow")