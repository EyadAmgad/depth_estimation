import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.midas_utils import load_midas_model, estimate_depth

st.set_page_config(page_title="Depth Estimation App", layout="centered")

st.title("🌄 Depth Estimation using MiDaS")
st.markdown("Upload an image and get the estimated depth map using a pre-trained MiDaS model.")

# Load the MiDaS model only once
@st.cache_resource
def load_model():
    return load_midas_model()

model, transform, device = load_model()

# Image uploader
uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run depth estimation
    with st.spinner("Estimating depth..."):
        depth_map = estimate_depth(image, model, transform, device)
    
    # Normalize and convert depth map for display
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

    st.image(depth_colored, caption="Estimated Depth Map", use_column_width=True)
else:
    st.info("👆 Upload an image to get started.")

st.markdown("---")
st.caption("Model: MiDaS (DPT_Large). Built with ❤️ by Eyad.")
