<img width="1280" height="675" alt="Screenshot 2026-06-28 at 11 04 04 AM" src="https://github.com/user-attachments/assets/2f37391f-f0ca-4611-8e4e-28ede10cfaec" /># 📸 DxOMARK-Style AI Image Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dxomark-ai-analyser.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An automated, interactive **Image Quality Assessment (IQA)** pipeline that combines **Deep Learning scene recognition** with traditional **Computer Vision algorithms** to benchmark camera and sensor performance.

👉 **[Try the Live Application Here!](https://dxomark-ai-analyser.streamlit.app/)**

---

#  App Output & Analytics

<img width="1280" height="675" alt="Screenshot 2026-06-28 at 11 04 04 AM" src="https://github.com/user-attachments/assets/86ffd57f-10ac-4f9d-a4e3-8ca307e5eb86" />



---

#  Project Overview

Commercial smartphone cameras are aggressively tuned by their Image Signal Processors (ISPs). To objectively measure their hardware capabilities, professional labs like **DxOMARK** rely on standardized test charts and rigorous mathematical evaluations.

This project democratizes that process.

**DxOMARK-Style AI Image Analyzer** is a hybrid Computer Vision application designed to automatically assess the technical quality of an image. Rather than relying solely on subjective human preference, it computes the raw physical performance of the camera's sensor and lens arrangement, producing an objective quality score out of **100**.

---

# 🔬 Deep Dive: Evaluation Metrics

The application follows a **multi-stage Image Quality Assessment (IQA) pipeline**. It first determines the image context using AI before applying mathematical Computer Vision algorithms to estimate camera performance.

---

## 1️⃣ AI Scene & Contextual Classification

Before evaluating image quality, the system identifies **what kind of image it is analyzing**. Different quality metrics are applicable for standardized calibration charts versus natural scenes.

### Technology

- **PyTorch**
- **Torchvision**
- **MobileNetV3 Small (Pretrained)**

### Process

- Performs image classification.
- Predicts the most likely scene category.
- Outputs a confidence score.
- Automatically routes the image into one of two pipelines:
  - **Standardized Calibration Target**
  - **Natural Aesthetic Scene**

---

## 2️⃣ Sharpness & Acutance (Focus Quality)

Sharpness indicates how well the camera captures edge detail.

Soft images often result from:

- Lens imperfections
- Missed autofocus
- Motion blur
- Atmospheric distortion

### Mathematical Method

The application computes the **Variance of the Laplacian** on the grayscale image.

The Laplacian emphasizes rapid intensity changes (edges). Images with many strong edges exhibit higher variance.

### Output

- Raw Laplacian Variance
- Normalized Sharpness Score (0–100)

Higher values indicate better focus and edge clarity.

---

## 3️⃣ Noise Control / Signal-to-Noise Ratio (SNR)

Every digital camera sensor introduces electronic noise, especially under low-light conditions or high ISO.

### Mathematical Method

The system samples a relatively uniform region near the image center.

It calculates:

- Mean Pixel Intensity (Signal)
- Standard Deviation (Noise)

The Signal-to-Noise Ratio is then computed as:

```text
SNR(dB) = 20 × log10(Mean / StandardDeviation)
```

### Interpretation

| SNR (dB) | Quality |
|----------|---------|
| >35 dB | Excellent |
| 30–35 dB | Very Good |
| 20–30 dB | Acceptable |
| <20 dB | Noticeable Noise |

Higher SNR indicates cleaner images.

---

## 4️⃣ Exposure Balance & Dynamic Range

Dynamic Range measures the camera's ability to preserve details in both highlights and shadows.

### Mathematical Method

The luminance histogram is analyzed.

Pixels are categorized as:

- **Blown Highlights:** Luma ≥ 254
- **Crushed Shadows:** Luma ≤ 2

The system computes the percentage of clipped pixels.

### Output

- Highlight Clipping
- Shadow Clipping
- Exposure Score

An ideal image has:

- Mid-tone average around **122**
- Minimal clipping
- Balanced histogram

---

#  Final Quality Score

Each metric contributes to an overall quality rating.

The final score combines:

- AI Scene Recognition
- Sharpness
- Signal-to-Noise Ratio
- Exposure Balance

The resulting **DxOMARK-style Quality Score** ranges from **0–100**.

---

#  Tech Stack

### Frontend

- Streamlit

### Computer Vision

- OpenCV
- NumPy

### Deep Learning

- PyTorch
- Torchvision
- MobileNetV3 Small

---

#  Local Installation

Clone the repository:

```bash
git clone https://github.com/Pranav080405/dxoMarkAI_analyser.git
cd dxoMarkAI_analyser
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

#  Repository Structure

```text
dxoMarkAI_analyser/
│
├── app.py                   # Streamlit dashboard
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── classifier.py        # MobileNetV3 inference engine
    └── metrics.py           # Image quality calculations
```

---

#  Future Roadmap

###  FastAPI Backend

Separate the heavy PyTorch and OpenCV inference into a dedicated REST API for improved scalability and deployment.

###  Automated PDF Reports

Generate publication-ready reports summarizing:

- Sharpness
- Noise
- Exposure
- Overall Quality Score

using LaTeX-based templates.

###  Color Accuracy (ΔE)

Implement automatic ColorChecker detection and compute **CIELAB Delta-E** color accuracy measurements.

---

#  Features

- AI-powered Scene Classification
- Sharpness Analysis
- Signal-to-Noise Ratio Estimation
- Dynamic Range Evaluation
- Automatic Image Quality Scoring
- Interactive Streamlit Dashboard
- Lightweight MobileNetV3 Backbone
- Real-time Computer Vision Metrics

