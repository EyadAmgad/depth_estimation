import cv2

def resize_frame(frame, size=(640, 480)):
    return cv2.resize(frame, size)

def pad_to_square(image, fill=0):
    height, width = image.shape[:2]
    size = max(height, width)
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
