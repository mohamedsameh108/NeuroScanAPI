import cv2
import numpy as np
from PIL import Image
import io

def image_processing(img):
    image_data = img.read()
    image = Image.open(io.BytesIO(image_data))
    img = np.array(image)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224,224))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
    img_clahe = clahe.apply(img_resized)
    norm_img = cv2.normalize(img_clahe, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    final_img = np.expand_dims(norm_img, axis=0)
    return final_img
    
