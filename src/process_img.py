import numpy as np
import torch
import cv2
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_image(pil_img):
    img = pil_img.convert('L')
    img = np.array(img)

    # Resize image to 32 height and/or 128 width if larger
    h, w = img.shape

    # The axes for cv2 image dimensions are swapped
    if w > 128:
        img = cv2.resize(img, (128, h), interpolation=cv2.INTER_AREA)
        w = 128

    if h > 32:
        img = cv2.resize(img, (w, 32), interpolation=cv2.INTER_AREA)
        h = 32

    # Add padding
    if h < 32 or w < 128:
        add_zeros_height = np.ones((32 - h, w)) * 255  # Pad height if needed
        add_zeros_width = np.ones((32, 128 - w)) * 255  # Pad width if needed

        img = np.concatenate((img, add_zeros_height), axis=0) if h < 32 else img
        img = np.concatenate((img, add_zeros_width), axis=1) if w < 128 else img

    # Normalize image
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img).type(torch.float32)