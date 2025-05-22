import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from model.midas_utils import load_midas_model
from model.depth_estimator import estimate_depth

# Ensure output folder exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load MiDaS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Large"  # or "MiDaS_small"
model, transform = load_midas_model(model_type=model_type)
model.to(device)
# Open webcam
cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20

# Define output video writer (side-by-side: webcam + depth map)
video_path = os.path.join(output_dir, "depth_output.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_path, fourcc, fps, (width * 2, height))

print("[INFO] Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate depth
    depth_map = estimate_depth(frame, model, transform)

    # Normalize depth map to 0-255 and apply colormap
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    depth_colored = cv2.resize(depth_colored, (width, height))

    # Combine webcam and depth map
    combined = cv2.hconcat([frame, depth_colored])

    # Show live view
    cv2.imshow("Depth Estimation", combined)

    # Save frame to video
    out.write(combined)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Video saved to: {video_path}")
