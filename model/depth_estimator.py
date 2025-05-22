import cv2
import torch
import torch.nn.functional as F

def estimate_depth(frame, model, transform):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img)
    print("Before unsqueeze, shape:", input_tensor.shape)

    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
    print("After unsqueeze, shape:", input_tensor.shape)

    try:
        with torch.no_grad():
            prediction = model(input_tensor)

            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        return prediction.cpu().numpy()

    except Exception as e:
        print("Error during depth estimation:", e)
        return None
