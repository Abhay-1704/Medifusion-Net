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

# === Load model with custom objects ===
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

def is_image_file(filename: str) -> bool:
    """Check if file is a valid image format"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in valid_extensions

def extract_images_from_zip(zip_file) -> List[tuple]:
    """Extract image files from uploaded zip file"""
    extracted_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir() and is_image_file(file_info.filename):
                    try:
                        zip_ref.extract(file_info, temp_dir)
                        extracted_path = os.path.join(temp_dir, file_info.filename)
                        
                        # Read the image
                        img = Image.open(extracted_path).convert("RGB")
                        extracted_files.append((file_info.filename, img))
                    except Exception as e:
                        st.warning(f"Could not process {file_info.filename}: {str(e)}")
    
    return extracted_files

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_single_image(img: Image.Image) -> Dict[str, Any]:
    """Make prediction for a single image"""
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
        'confidence': confidence,
        'epistemic_uncertainty': epistemic_unc,
        'aleatoric_uncertainty': aleatoric_unc,
        'prediction_index': pred_idx,
        'all_predictions': main_preds,
        'processed_image': img_batch
    }

def generate_gradcam(img_batch: np.ndarray, pred_idx: int) -> np.ndarray:
    """Generate Grad-CAM visualization"""
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
        
        # Convert processed image back to displayable format
        img_array = img_batch[0]
        superimposed = cv2.addWeighted(np.uint8(img_array * 255), 0.6, heatmap_colored, 0.4, 0)
        
        return superimposed
        
    except Exception as e:
        st.error(f"Failed to generate Grad-CAM: {str(e)}")
        return None

def display_single_prediction(filename: str, img: Image.Image, result: Dict[str, Any], show_gradcam: bool = True):
    """Display prediction results for a single image"""
    st.subheader(f"📋 Results for: {filename}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption=f"Original: {filename}", use_column_width=True)
        
    with col2:
        st.write(f"**🎯 Predicted Class:** {result['predicted_class']}")
        st.write(f"**📊 Confidence:** {result['confidence']:.4f}")
        st.write(f"**🔍 Epistemic Uncertainty:** {result['epistemic_uncertainty']:.4f}")
        st.write(f"**⚡ Aleatoric Uncertainty:** {result['aleatoric_uncertainty']:.4f}")
        
        # Show confidence distribution
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
    """Create summary DataFrame for batch processing"""
    summary_data = []
    
    for filename, result in zip(filenames, results):
        summary_data.append({
            'Filename': filename,
            'Predicted Class': result['predicted_class'],
            'Confidence': f"{result['confidence']:.4f}",
            'Epistemic Uncertainty': f"{result['epistemic_uncertainty']:.4f}",
            'Aleatoric Uncertainty': f"{result['aleatoric_uncertainty']:.4f}"
        })
    
    return pd.DataFrame(summary_data)

# === Streamlit UI ===
st.title("🩺 Carelens AI - Advanced Chest X-Ray Classifier")
st.markdown("Upload single images, multiple files, a ZIP folder, or capture with camera for batch processing")

# === Upload Options ===
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
        uploaded_files = [uploaded_files]  # Convert to list for consistency

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

# === Processing ===
if uploaded_files or zip_file or camera_image:
    
    # Handle camera image
    if camera_image:
        st.info("📸 Processing camera image...")
        try:
            img = Image.open(camera_image).convert("RGB")
            images_to_process = [("camera_capture.jpg", img)]
        except Exception as e:
            st.error(f"❌ Could not process camera image: {str(e)}")
            st.stop()
    
    # Handle ZIP file
    elif zip_file:
        st.info("📦 Extracting images from ZIP file...")
        extracted_images = extract_images_from_zip(zip_file)
        
        if not extracted_images:
            st.error("❌ No valid image files found in the ZIP archive")
            st.stop()
        
        st.success(f"✅ Found {len(extracted_images)} image(s) in the ZIP file")
        
        # Process extracted images
        images_to_process = extracted_images
        
    else:
        # Handle direct file uploads
        images_to_process = []
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                images_to_process.append((uploaded_file.name, img))
            except Exception as e:
                st.error(f"❌ Could not process {uploaded_file.name}: {str(e)}")
    
    if images_to_process:
        # === Batch Processing Options ===
        st.subheader("⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_individual_results = st.checkbox("Show individual results", value=True)
        with col2:
            show_gradcam = st.checkbox("Generate Grad-CAM visualizations", value=len(images_to_process) <= 5)
        
        if len(images_to_process) > 10:
            st.warning("⚠️ Processing many images may take some time. Consider disabling Grad-CAM for faster results.")
        
        # === Process Images ===
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
            
            # === Batch Summary ===
            if len(results) > 1:
                st.subheader("📊 Batch Processing Summary")
                
                summary_df = create_batch_summary(results, filenames)
                st.dataframe(summary_df, use_container_width=True)
                
                # === Download Results ===
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name="medifusion_batch_results.csv",
                    mime="text/csv"
                )
                
                # === Class Distribution ===
                st.subheader("📈 Class Distribution")
                class_counts = summary_df['Predicted Class'].value_counts()
                st.bar_chart(class_counts)
                
                # === Confidence Statistics ===
                st.subheader("📊 Confidence Statistics")
                confidence_values = [float(conf) for conf in summary_df['Confidence']]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Confidence", f"{np.mean(confidence_values):.4f}")
                with col2:
                    st.metric("Min Confidence", f"{np.min(confidence_values):.4f}")
                with col3:
                    st.metric("Max Confidence", f"{np.max(confidence_values):.4f}")

# === Sidebar with Information ===
st.sidebar.title("ℹ️ Information")
st.sidebar.markdown("""
### Supported Upload Methods:
- **Single Image**: Upload one X-ray image
- **Multiple Images**: Select multiple X-ray images
- **ZIP Folder**: Upload a ZIP file containing X-ray images
- **Camera Capture**: Take a picture using your device camera

### Supported Formats:
- PNG, JPG, JPEG, BMP, TIFF

### Features:
- 🎯 AI-powered chest X-ray classification
- 📊 Confidence scoring with uncertainty quantification
- 🔥 Grad-CAM visualization for explainability
- 📈 Batch processing with summary statistics
- 💾 Downloadable results in CSV format
- 📸 Real-time camera capture

### Tips:
- For large batches, consider disabling Grad-CAM for faster processing
- ZIP files should contain images directly or in subdirectories
- Results include both epistemic and aleatoric uncertainty measures
- Camera capture works on devices with camera access
""")

# === Footer ===
st.markdown("---")
st.markdown("🩺 **MediFusion AI** - Advanced Medical Image Analysis | Built with Streamlit & TensorFlow")
