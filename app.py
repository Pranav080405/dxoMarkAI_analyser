import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Corrected Imports
from src.classifier import SceneClassifier
from src.metrics import analyze_sharpness, analyze_noise_snr, analyze_exposure_and_dynamic_range

st.set_page_config(page_title="DxOMARK-Style Lab Analyzer", layout="wide")

st.title("📸 Automated DxOMARK-Style Image Lab")
st.write("An interactive tool demonstrating how Computer Vision and Machine Learning benchmarks image quality.")

# Initialize the AI Model
@st.cache_resource
def load_ai_model():
    return SceneClassifier()

classifier = load_ai_model()

# Sidebar Setup
st.sidebar.header("🔬 Target Analyzer Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    pil_image = Image.open(uploaded_file).convert("RGB")
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Layout Split
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Target Image under Test")
        st.image(pil_image, use_column_width=True)
        
        # Pipeline Step 1: AI Object Recognition
        with st.spinner("AI Processing Scene Context..."):
            scene_type, profile, confidence = classifier.predict(pil_image)
            
        st.info(f"**AI Classification Result:** {scene_type}  \n**Pipeline Profiler:** Assigned to **{profile}** (Confidence: {confidence:.2%})")

    with col2:
        st.subheader("📊 Objective Evaluation Metrics")
        
        # Compute Traditional CV Hardware Metrics
        sharp_score, raw_lap = analyze_sharpness(opencv_image)
        noise_score, raw_snr = analyze_noise_snr(opencv_image)
        exp_score, raw_bright, raw_clip = analyze_exposure_and_dynamic_range(opencv_image)
        
        # Aggregate Final Benchmark Rating (DxOMARK Style Overall Weighting)
        dxomark_overall = int((sharp_score * 0.4) + (noise_score * 0.3) + (exp_score * 0.3))
        
        st.metric(label="Overall Engineering Metric Score", value=f"{dxomark_overall} / 100")
        st.progress(dxomark_overall / 100)
        
        st.markdown("---")
        
        # Parameter Breakdown
        st.markdown(f"### 🔍 Sharpness / Acutance: **{sharp_score:.1f}/100**")
        st.caption(f"Raw Laplacian Edge Variance Score: `{raw_lap:.2f}`. This evaluates lens clarity and high-frequency sensor resolution mapping.")
        
        st.markdown(f"### 🛡️ Noise Control (SNR): **{noise_score:.1f}/100**")
        st.caption(f"Calculated Center Signal-to-Noise Ratio: `{raw_snr:.2f} dB`. Higher decibels indicate lower pixel fluctuation variance in flat regions.")
        
        st.markdown(f"### 🌤️ Exposure & Tonal Range: **{exp_score:.1f}/100**")
        st.caption(f"Mean Luma Intensity: `{raw_bright:.1f}` | Highlight/Shadow Clipping Factor: `{raw_clip:.2%}`.")

        # Educational Framework Panel
        with st.expander("💡 Learning Lab: How to read these results?"):
            st.markdown("""
            * **Why did the AI classify this?** Different testing parameters apply to test charts versus natural scenery. True hardware metrics utilize charts, while human aesthetic preferences guide real-world analysis.
            * **What is Sharpness measuring?** It utilizes edge gradient processing. Lower scores usually point to lens softness, atmospheric diffusion, or motion defocus.
            * **What does the SNR value mean?** If the decibel level falls below $20\\text{dB}$, the image will appear visibly grainy due to sensor thermal limitations or aggressive gain calibration.
            """)
else:
    st.info("👈 Please upload an image in the sidebar interface to begin running the automated analytics matrix.")