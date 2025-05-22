import numpy as np
import cv2

def visualize_depth(depth_map):
    # Normalize to 0–255 range for display
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_8bit = depth_normalized.astype(np.uint8)

    # Apply a color map (e.g., magma, jet, etc.)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)

    return depth_colored
