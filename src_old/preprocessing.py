import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from src.logger import initialize_logger
from src.config import IMAGE_WIDTH, IMAGE_HEIGHT

log = initialize_logger(__name__)


def _resize_and_pad(binary_matrix: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = binary_matrix.shape
    calculated_aspect = w / max(1, h)
    target_aspect = target_w / max(1, target_h)

    if calculated_aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / max(1e-6, calculated_aspect))
    else:
        new_h = target_h
        new_w = int(target_h * calculated_aspect)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
    resized_matrix = cv2.resize(binary_matrix, (new_w, new_h), interpolation=interpolation)

    padded_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    padded_canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_matrix

    normalized_tensor = padded_canvas.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(normalized_tensor), axis=-1)


def execute_morphological_preprocessing_from_array(
    raw_img: np.ndarray,
    target_w: int = IMAGE_WIDTH,
    target_h: int = IMAGE_HEIGHT,
) -> Optional[np.ndarray]:
    """
    Word-level preprocessing for model inference.
    Uses ADAPTIVE binarization (recognition path).
    """
    try:
        if raw_img is None or raw_img.size == 0:
            return None
        if len(raw_img.shape) == 3:
            raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            raw_gray = raw_img

        blurred = cv2.GaussianBlur(raw_gray, (5, 5), 0)

        # Adaptive binarization only for recognition preprocessing
        binary_inverse = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            65,
            19,
        )

        cleaned_binary = cv2.medianBlur(binary_inverse, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_matrix = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel)

        return _resize_and_pad(processed_matrix, target_w, target_h)

    except Exception as exception_trace:
        log.error("Array preprocessing failed: %s", str(exception_trace))
        return None


def execute_morphological_preprocessing(
    image_filepath: str,
    target_w: int = IMAGE_WIDTH,
    target_h: int = IMAGE_HEIGHT,
) -> Optional[np.ndarray]:
    """
    Backward-compatible file-path wrapper used by training/inference code.
    """
    try:
        # Image greyscale
        raw_img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Source image missing or corrupted: {image_filepath}")
        return execute_morphological_preprocessing_from_array(raw_img, target_w, target_h)
    except Exception as exception_trace:
        log.error("Preprocessing aborted for %s: %s", image_filepath, str(exception_trace))
        return None


def perform_otsu_binarization(page_gray: np.ndarray) -> np.ndarray:
    """
    Otsu binarization for segmentation path.
    """
    _, bw = cv2.threshold(page_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels = np.count_nonzero(bw == 255)
    black_pixels = np.count_nonzero(bw == 0)
    if white_pixels > black_pixels:
        bw = cv2.bitwise_not(bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw


def _extract_runs(active_rows: np.ndarray, min_len: int = 6) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None

    for idx, flag in enumerate(active_rows):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx))
            start = None

    if start is not None and len(active_rows) - start >= min_len:
        runs.append((start, len(active_rows)))

    return runs


def _find_horizontal_seam(
    cost_map: np.ndarray,
    y_hint: int,
    band_half_height: int,
    penalty: float = 0.02,
) -> np.ndarray:
    """
    Dynamic-programming seam path from left->right near y_hint.
    Lower cost is preferred (whitespace valleys).
    """
    h, w = cost_map.shape
    if h == 0 or w == 0:
        return np.zeros((0,), dtype=np.int32)

    y_hint = int(np.clip(y_hint, 0, h - 1))
    y_min = max(0, y_hint - band_half_height)
    y_max = min(h - 1, y_hint + band_half_height)

    dp = np.full((h, w), np.inf, dtype=np.float32)
    parent = np.full((h, w), -1, dtype=np.int32)

    for y in range(y_min, y_max + 1):
        dp[y, 0] = cost_map[y, 0] + penalty * abs(y - y_hint)

    for x in range(1, w):
        for y in range(y_min, y_max + 1):
            best_prev_cost = np.inf
            best_prev_y = y
            for prev_y in (y - 1, y, y + 1):
                if prev_y < y_min or prev_y > y_max:
                    continue
                prev_cost = dp[prev_y, x - 1]
                if prev_cost < best_prev_cost:
                    best_prev_cost = prev_cost
                    best_prev_y = prev_y

            dp[y, x] = cost_map[y, x] + penalty * abs(y - y_hint) + best_prev_cost
            parent[y, x] = best_prev_y

    end_segment = dp[y_min:y_max + 1, w - 1]
    end_y = y_min + int(np.argmin(end_segment))

    seam = np.zeros((w,), dtype=np.int32)
    seam[w - 1] = end_y
    for x in range(w - 1, 0, -1):
        seam[x - 1] = parent[seam[x], x]

    return seam


def segment_lines_with_seam_carving(page_gray: np.ndarray, page_ink: np.ndarray) -> List[Dict[str, np.ndarray]]:
    """
    Line segmentation using seam carving boundaries between adjacent text runs.
    """
    h, _ = page_ink.shape
    projection = np.sum(page_ink > 0, axis=1).astype(np.float32)
    projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 31), 0).reshape(-1)

    threshold = max(2.0, 0.08 * float(np.max(projection)))
    active_rows = projection >= threshold
    runs = _extract_runs(active_rows, min_len=max(6, h // 200))

    if not runs:
        return []

    background = (page_ink == 0).astype(np.uint8)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 3)
    cost_map = (1.0 / (dist + 1.0)).astype(np.float32)
    cost_map += (page_ink > 0).astype(np.float32) * 4.0

    boundaries: List[int] = []
    for idx in range(len(runs) - 1):
        upper = runs[idx]
        lower = runs[idx + 1]
        y_hint = (upper[1] + lower[0]) // 2
        gap = max(10, lower[0] - upper[1])
        seam = _find_horizontal_seam(cost_map, y_hint=y_hint, band_half_height=max(12, gap))
        if seam.size == 0:
            continue
        boundary = int(np.median(seam))
        if 2 < boundary < h - 2:
            boundaries.append(boundary)

    boundaries = sorted(boundaries)
    filtered_boundaries: List[int] = []
    for boundary in boundaries:
        if not filtered_boundaries or boundary - filtered_boundaries[-1] >= 4:
            filtered_boundaries.append(boundary)

    split_points = [0] + filtered_boundaries + [h]
    line_regions: List[Dict[str, np.ndarray]] = []

    for idx in range(len(split_points) - 1):
        top = split_points[idx]
        bottom = split_points[idx + 1]
        if bottom - top < 8:
            continue

        line_ink = page_ink[top:bottom, :]
        if np.count_nonzero(line_ink) < 30:
            continue

        line_gray = page_gray[top:bottom, :]
        line_regions.append(
            {
                "gray": line_gray,
                "ink": line_ink,
                "top": int(top),
                "bottom": int(bottom),
            }
        )

    return line_regions


def segment_document_into_word_images(image_filepath: str) -> Optional[Dict[str, Any]]:
    """
    Segmentation path uses Otsu for page binarization.
    """
    try:
        page_gray = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if page_gray is None:
            raise FileNotFoundError(f"Source image missing or corrupted: {image_filepath}")

        # Otsu for segmentation
        page_ink = perform_otsu_binarization(page_gray)
        line_regions = segment_lines_with_seam_carving(page_gray, page_ink)

        words_by_line: List[List[np.ndarray]] = []
        line_boxes: List[Tuple[int, int]] = []

        for line_region in line_regions:
            line_words = segment_words_scale_space(line_region["gray"], line_region["ink"])
            if line_words:
                words_by_line.append(line_words)
                line_boxes.append((int(line_region["top"]), int(line_region["bottom"])))

        total_words = sum(len(words) for words in words_by_line)

        return {
            "words_by_line": words_by_line,
            "word_count": total_words,
            "line_count": len(words_by_line),
            "line_boxes": line_boxes,
            "page_ink": page_ink,
        }

    except Exception as exception_trace:
        log.error("Document segmentation failed for %s: %s", image_filepath, str(exception_trace))
        return None


def segment_words_scale_space(
    line_gray: np.ndarray,
    line_ink: np.ndarray,
    base_kernel: Tuple[int, int] = (25, 5),
) -> List[np.ndarray]:
    """
    Segmentation path keeps Otsu on blurred line mask.
    """
    if line_gray is None or line_ink is None or line_ink.size == 0:
        return []

    if len(line_gray.shape) == 3:
        line_gray = cv2.cvtColor(line_gray, cv2.COLOR_BGR2GRAY)
    if len(line_ink.shape) == 3:
        line_ink = cv2.cvtColor(line_ink, cv2.COLOR_BGR2GRAY)

    line_h, line_w = line_ink.shape

    kw = max(base_kernel[0], int(line_w * 0.06))
    kh = max(base_kernel[1], int(line_h * 0.35))
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1

    blurred = cv2.GaussianBlur(line_ink, (kw, kh), 0)
    _, merged_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = max(40, int(0.0025 * line_h * line_w))
    min_height = max(6, int(0.20 * line_h))
    boxes: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or h < min_height or w < 4:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    word_crops: List[np.ndarray] = []
    for x, y, w, h in boxes:
        pad_x = max(2, w // 15)
        pad_y = max(2, h // 8)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(line_w, x + w + pad_x)
        y1 = min(line_h, y + h + pad_y)

        crop = line_gray[y0:y1, x0:x1]
        if crop.size > 0:
            word_crops.append(crop)

    return word_crops