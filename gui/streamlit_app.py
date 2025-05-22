import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np

from model.midas_utils import load_midas_model
from model.depth_estimator import estimate_depth
from utils.visualization import visualize_depth
st.title("Real-Time Depth Estimation")
st.markdown("Using MiDaS model and your webcam 📷")

model, transform = load_midas_model()

cap = cv2.VideoCapture(0)

stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame.")
        break

    depth = estimate_depth(frame, model, transform)
    depth_colored = visualize_depth(depth)

    rgbd_combined = cv2.hconcat([frame, depth_colored])
    rgbd_combined = cv2.cvtColor(rgbd_combined, cv2.COLOR_BGR2RGB)

    stframe.image(rgbd_combined, channels="RGB")

cap.release()
