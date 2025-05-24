import torch
import torchvision.transforms as T
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image

def load_midas_model():
    model_type = "DPT_Large"  # Other options: DPT_Hybrid, MiDaS_small
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform, device

def estimate_depth(image, model, transform, device):
    img = transform(image).to(device)
    with torch.no_grad():
        prediction = model(img.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # PIL uses (width, height)
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map
