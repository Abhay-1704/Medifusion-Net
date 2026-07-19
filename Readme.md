# MediFusion-Net

**A Novel Hybrid Deep Learning Framework for Medical Image Classification with Comprehensive Uncertainty Quantification and Interpretability.**

> **Published in Biomedical Signal Processing and Control (Elsevier)**  
> Volume 126, Part B, 15 October 2026, 110785.  
> [Read the full publication on ScienceDirect](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)

---

## 1. Overview

MediFusion-Net (also referred to as Carelens AI in the interface) is a comprehensive, production-ready medical image analysis pipeline designed to bridge the gap between high-performance deep learning and clinical reliability. In medical diagnostics, a model's accuracy is insufficient without measurable trust and interpretability. 

To address this, MediFusion-Net introduces a hybrid architectural framework that fuses **Convolutional Neural Networks (CNNs)** for localized feature extraction with **Vision Transformers (ViTs)** for global anatomical reasoning. The framework explicitly addresses the critical challenge of AI overconfidence by integrating **Bayesian Uncertainty Quantification**, effectively decomposing model doubt into Epistemic (model-based) and Aleatoric (data-based) uncertainty. 

## 2. Core Architecture

The MediFusion-Net backbone is highly customized to handle the complexities of radiological scans:

*   **Hybrid CNN-ViT Integration:** The model leverages a dual-stream approach. The CNN pathway extracts fine-grained, shift-invariant local textures (crucial for identifying micro-lesions), while the ViT pathway captures long-range spatial dependencies across the entire anatomy.
*   **Squeeze-and-Excitation (SE) Blocks:** Channel-wise attention mechanisms are injected into the CNN layers to dynamically re-calibrate feature maps, suppressing irrelevant background noise and emphasizing critical pathological markers.
*   **Adaptive Focal Loss:** Medical datasets inherently suffer from severe class imbalances (e.g., rare pathologies vs. normal scans). The network utilizes Adaptive Focal Loss to dynamically scale the gradient updates, forcing the model to focus on hard-to-classify and underrepresented examples.
*   **Multi-Organ Registry:** The architecture dynamically loads specialized weights for different anatomical regions (Lungs, Bone, Kidney) on demand, preserving modularity.

## 3. Bayesian Uncertainty Quantification

Unlike standard softmax outputs, which often produce falsely confident probabilities, MediFusion-Net implements a Bayesian probabilistic layer utilizing Monte Carlo (MC) Dropout. This allows the system to generate a predictive distribution rather than a point estimate.

The uncertainty is mathematically decomposed into two actionable metrics for clinicians:
1.  **Epistemic Uncertainty (Model Uncertainty):** Arises from a lack of training data. High epistemic uncertainty indicates the model is encountering a novel or out-of-distribution anatomical pattern. This signals a need for more training data or immediate radiologist review.
2.  **Aleatoric Uncertainty (Data Uncertainty):** Arises from inherent noise in the image itself (e.g., poor scan resolution, motion artifacts, or inherently ambiguous overlapping structures). High aleatoric uncertainty indicates that even a perfect model would struggle to classify the image.

## 4. Clinical Analysis & Interpretability Suite

The repository includes a comprehensive, 9-module Streamlit clinical dashboard that serves as both an inference interface and a deep data-mining analysis suite.

### Diagnostic Tools
*   **Single Image Inference & Grad-CAM:** Real-time classification accompanied by Gradient-weighted Class Activation Mapping (Grad-CAM) to visually highlight the regions of the X-ray driving the prediction.
*   **Batch Classification:** Processes ZIP archives of patient scans, generating aggregated statistical reports and CSV exports.

### Data Mining & Quality Control
*   **Image Quality Checker:** Automated pre-inference screening that evaluates Laplacian variance (sharpness) and grayscale distribution (brightness) to reject substandard scans.
*   **Outlier Detection:** Interquartile Range (IQR) based statistical isolation of anomalous scans within a batch.
*   **Association Rules:** Apriori-based data mining to discover concurrent clinical tags and patterns across large batches.
*   **k-NN Similarity Finder:** Extracts a 64-bin normalized intensity histogram (Euclidean feature-space fingerprint) to retrieve the visually closest historical cases, assisting radiologists via cross-referencing.

### Validation & Reliability Dashboards
*   **Uncertainty Quadrant Plot:** Plots Epistemic vs. Aleatoric uncertainty on a 2D scatter plot, sorting predictions into 4 actionable quadrants (e.g., Confident & Clear vs. Doubly Uncertain).
*   **Bias & Variance Explorer:** Performs per-class stability analysis by evaluating the standard deviation of confidence scores. This identifies which pathologies the model underfits (High Bias) and which it predicts inconsistently (High Variance).

---

## 5. Installation & Setup

### Prerequisites
The environment requires Python 3.9+ and the following core libraries:
*   TensorFlow 2.10+
*   Streamlit
*   OpenCV
*   Scikit-Learn
*   Pandas / NumPy / Matplotlib

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MediFusion-Net.git
   cd MediFusion-Net
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the pre-trained anatomical weights are correctly mapped in the `models/` directory:
   *   `models/Lungs/`
   *   `models/Bone/`
   *   `models/Kidney/`

### Running the System
To launch the interactive clinical dashboard:
```bash
streamlit run Interface.py
```

---

## 6. Citation

If you utilize MediFusion-Net, its architecture, or its uncertainty decomposition methodologies in your research, please cite the official publication:

```bibtex
@article{singh2026medifusion,
  title={MediFusion-Net: A novel hybrid deep learning framework for medical image classification with comprehensive uncertainty quantification and interpretability},
  author={Singh, Abhay Pratap and Patel, R.B.},
  journal={Biomedical Signal Processing and Control},
  volume={126},
  pages={110785},
  year={2026},
  publisher={Elsevier}
}
```

---
*Disclaimer: This software is designed for research, educational, and investigational purposes only. It is not approved by regulatory bodies for direct clinical diagnosis without human radiologist supervision.*
