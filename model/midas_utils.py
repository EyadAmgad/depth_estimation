import torch

def load_midas_model(model_type="DPT_Large"):
    # Load model from Torch Hub
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    # Load transform properly based on model type
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform
