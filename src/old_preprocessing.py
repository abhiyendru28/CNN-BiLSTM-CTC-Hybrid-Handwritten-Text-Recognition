import cv2
import numpy as np
from typing import Optional
from src.logger import initialize_logger
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT

log = initialize_logger(__name__)

def execute_morphological_preprocessing(image_filepath: str, target_w: int = IMAGE_WIDTH, target_h: int = IMAGE_HEIGHT) -> Optional[np.ndarray]:
    """
    Executes Grayscale Conversion, Adaptive Inverse Binarization, Dilation, and Padding.
    Matches the exact morphological preprocessing pipeline described in the literature.
    """
    try:
        # Load directly into grayscale to bypass RGB channel processing
        raw_img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Source image missing or corrupted: {image_filepath}")
        raw_img = cv2.convertScaleAbs(raw_img, alpha=1.15, beta=10)
        raw_img = cv2.fastNlMeansDenoising(raw_img, None, 10, 7, 21)

        # 1. Adaptive Binarization (Inverse Mode) 
        # Dynamically threshold based on local 11x11 neighborhood blocks to combat uneven lighting
        binary_inverse = cv2.adaptiveThreshold(
            raw_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 2. Morphological Dilation 
        # Slide a 2x2 structural kernel to connect broken handwriting strokes
        structuring_kernel = np.ones((2, 2), np.uint8)
        dilated_matrix = cv2.dilate(binary_inverse, structuring_kernel, iterations=1)

        # 3. Geometric Normalization and Padding
        # Maintain aspect ratio to prevent destructive stroke warping
        h, w = dilated_matrix.shape
        calculated_aspect = w / h
        target_aspect = target_w / target_h

        if calculated_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / calculated_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * calculated_aspect)

        interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        resized_matrix = cv2.resize(dilated_matrix, (new_w, new_h), interpolation=interpolation)
        
        # Construct a pitch-black canvas and paste the resized image in the absolute center
        padded_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        padded_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_matrix

        # Normalize 8-bit integer pixel intensities  to float32 bounds [0.0, 1.0]
        normalized_tensor = padded_canvas.astype(np.float32) / 255.0
        
        # Transpose to (Width, Height) format. Time-distributed processing in Keras 
        # reads the primary axis as the time sequence. Add channel dimension.
        final_tensor = np.expand_dims(np.transpose(normalized_tensor), axis=-1)
        
        return final_tensor

    except Exception as exception_trace:
        log.error(f"Preprocessing aborted for {image_filepath}: {str(exception_trace)}")
        return None