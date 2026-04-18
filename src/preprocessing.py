import cv2
import numpy as np
from typing import Optional
from src.logger import initialize_logger
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT

log = initialize_logger(__name__)

def execute_morphological_preprocessing(image_filepath: str, target_w: int = IMAGE_WIDTH, target_h: int = IMAGE_HEIGHT) -> Optional[np.ndarray]:
    """
    Optimized Grayscale Conversion, Blurring, Adaptive Inverse Binarization, 
    Noise Removal, Morphological Closing, and Padding.
    """
    try:
        raw_img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Source image missing or corrupted: {image_filepath}")
        
        # 1. Gaussian Blur
        blurred = cv2.GaussianBlur(raw_img, (5, 5), 0)

        # 2. Adaptive Binarization (Inverse Mode) 
        binary_inverse = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 65, 19
        )

        # 3. Median Blur
        cleaned_binary = cv2.medianBlur(binary_inverse, 3)

        # 4. Morphological Closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_matrix = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel)

        # 5. Geometric Normalization and Padding 
        h, w = processed_matrix.shape
        calculated_aspect = w / h
        target_aspect = target_w / target_h

        if calculated_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / calculated_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * calculated_aspect)

        # Failsafe for 0 dimension
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        resized_matrix = cv2.resize(processed_matrix, (new_w, new_h), interpolation=interpolation)
        
        padded_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        padded_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_matrix

        # 6. Normalize and Reshape for Keras/CTC
        normalized_tensor = padded_canvas.astype(np.float32) / 255.0
        final_tensor = np.expand_dims(np.transpose(normalized_tensor), axis=-1)
        
        return final_tensor

    except Exception as exception_trace:
        log.error(f"Preprocessing aborted for {image_filepath}: {str(exception_trace)}")
        print(f"Error: {exception_trace}")
        return None


