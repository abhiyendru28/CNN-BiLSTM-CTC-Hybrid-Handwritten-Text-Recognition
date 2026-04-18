import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename
from typing import List, Tuple

from src.architecture import compile_hybrid_network
from src.preprocessing import (
    execute_morphological_preprocessing,
    execute_morphological_preprocessing_from_array,
    perform_otsu_binarization,
    segment_document_into_word_images,
    segment_words_scale_space,
)
from src.inference_engine import execute_ctc_decoding, build_replication_lm_decoder
from src.config import MODEL_DIRECTORY, REPLICATION_MODE, STRICT_LM_DECODER

app = Flask(__name__)

upload_dir = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = upload_dir
os.makedirs(upload_dir, exist_ok=True)

_, model_inference = compile_hybrid_network()
weight_filepath = os.path.join(MODEL_DIRECTORY, "optimal_hybrid_weights.weights.h5")

if os.path.exists(weight_filepath):
    model_inference.load_weights(weight_filepath)
    print("Inference engine loaded successfully.")
else:
    message = f"Missing weights file: {weight_filepath}"
    if REPLICATION_MODE:
        raise FileNotFoundError(message)
    print("Warning:", message)

lm_decoder = build_replication_lm_decoder(required=STRICT_LM_DECODER)


@app.route("/")
def index() -> ResponseReturnValue:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> ResponseReturnValue:
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    raw_filename = file.filename
    if not raw_filename:
        return jsonify({"ok": False, "error": "No selected file"}), 400

    filename = secure_filename(raw_filename)
    if not filename:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    output_level = _normalize_output_level(
        request.form.get("output_level") or request.args.get("output_level")
    )

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Word-level mode: direct decode, no segmentation
    if output_level == "word":
        processed_tensor = execute_morphological_preprocessing(filepath)  # adaptive path
        if processed_tensor is None:
            return jsonify({"ok": False, "error": "Preprocessing failed"}), 500

        try:
            vis_matrix = np.transpose(processed_tensor[..., 0])  # (H, W)
            vis_img = (vis_matrix * 255.0).clip(0, 255).astype(np.uint8)
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
            cv2.imwrite(processed_path, vis_img)
            processed_image_url = url_for("static", filename=f"uploads/{processed_filename}")
        except Exception:
            processed_image_url = None

        batch_tensor = np.expand_dims(processed_tensor, axis=0)
        softmax_output = model_inference.predict(batch_tensor, verbose=0)
        decoded = execute_ctc_decoding(
            softmax_output,
            lm_decoder=lm_decoder,
            require_lm=STRICT_LM_DECODER,
        )

        if not decoded:
            return jsonify({"ok": False, "error": "Decoding failed"}), 500

        prediction_text = (decoded[0] or "").strip()
        if not prediction_text:
            return jsonify({"ok": False, "error": "Decoding failed"}), 500

        predicted_words = [token for token in prediction_text.split() if token]
        decoded_word_count = len(predicted_words)

        image_url = url_for("static", filename=f"uploads/{filename}")
        return jsonify(
            {
                "ok": True,
                "output_level": "word",
                "prediction": prediction_text,
                "paragraph_text": prediction_text,
                "paragraphs": [prediction_text],
                "lines": [prediction_text],
                "words_by_line": [predicted_words],
                "word_count": decoded_word_count,
                "decoded_word_count": decoded_word_count,
                "line_count": 1,
                "decoded_line_count": 1,
                "paragraph_count": 1,
                "segmentation_used": False,
                "image_url": image_url,
                "preprocessed_image_url": processed_image_url,
            }
        ), 200

    # Line-level mode: segmentation path (Otsu)
    if output_level == "line":
        line_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if line_gray is None:
            return jsonify({"ok": False, "error": "Could not read input image"}), 400

        line_ink = perform_otsu_binarization(line_gray)  # Otsu for segmentation
        line_words = segment_words_scale_space(line_gray, line_ink)

        words_by_line = [line_words] if line_words else []
        line_boxes = [(0, int(line_gray.shape[0]))] if line_words else []
        detected_word_count = len(line_words)
        detected_line_count = 1 if line_words else 0
        processed_vis = line_ink
    else:
        document = segment_document_into_word_images(filepath)
        if document is None:
            return jsonify({"ok": False, "error": "Document segmentation failed"}), 500

        words_by_line = document["words_by_line"]
        line_boxes = [tuple(box) for box in document.get("line_boxes", [])]
        detected_word_count = int(document["word_count"])
        detected_line_count = int(document["line_count"])
        processed_vis = document["page_ink"]

    if detected_word_count == 0:
        return jsonify({"ok": False, "error": "No words detected in the image"}), 400

    try:
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
        cv2.imwrite(processed_path, processed_vis)
        processed_image_url = url_for("static", filename=f"uploads/{processed_filename}")
    except Exception:
        processed_image_url = None

    flat_word_tensors = []
    line_to_batch_indices: List[List[int]] = []

    for line_words in words_by_line:
        indices: List[int] = []
        for word_img in line_words:
            word_tensor = execute_morphological_preprocessing_from_array(word_img)
            if word_tensor is None:
                indices.append(-1)
                continue
            flat_word_tensors.append(word_tensor)
            indices.append(len(flat_word_tensors) - 1)
        line_to_batch_indices.append(indices)

    if not flat_word_tensors:
        return jsonify({"ok": False, "error": "Word preprocessing failed"}), 500

    batch_tensor = np.stack(flat_word_tensors, axis=0)
    softmax_output = model_inference.predict(batch_tensor, verbose=0)
    decoded_batch = execute_ctc_decoding(
        softmax_output,
        lm_decoder=lm_decoder,
        require_lm=STRICT_LM_DECODER,
    )

    decoded_words_by_line: List[List[str]] = []
    decoded_lines: List[str] = []
    decoded_line_boxes: List[Tuple[int, int]] = []
    decoded_word_count = 0

    for line_idx, indices in enumerate(line_to_batch_indices):
        line_tokens: List[str] = []
        for idx in indices:
            if idx < 0 or idx >= len(decoded_batch):
                continue
            token = (decoded_batch[idx] or "").strip()
            if token:
                line_tokens.append(token)

        decoded_words_by_line.append(line_tokens)

        if line_tokens:
            decoded_word_count += len(line_tokens)
            decoded_lines.append(" ".join(line_tokens))
            if line_idx < len(line_boxes):
                top, bottom = line_boxes[line_idx]
                decoded_line_boxes.append((int(top), int(bottom)))

    paragraphs = _group_lines_into_paragraphs(decoded_lines, decoded_line_boxes)
    paragraph_text = "\n\n".join(paragraphs).strip()

    if not paragraph_text:
        paragraph_text = "\n".join(decoded_lines).strip()

    if not paragraph_text:
        return jsonify({"ok": False, "error": "Decoding failed"}), 500

    if output_level == "word":
        prediction_text = " ".join(
            token for line_tokens in decoded_words_by_line for token in line_tokens
        ).strip()
    elif output_level == "line":
        prediction_text = "\n".join(decoded_lines).strip()
    else:
        prediction_text = paragraph_text

    image_url = url_for("static", filename=f"uploads/{filename}")

    return jsonify(
        {
            "ok": True,
            "output_level": output_level,
            "prediction": prediction_text,
            "paragraph_text": paragraph_text,
            "paragraphs": paragraphs,
            "lines": decoded_lines,
            "words_by_line": decoded_words_by_line,
            "word_count": detected_word_count,
            "decoded_word_count": decoded_word_count,
            "line_count": detected_line_count,
            "decoded_line_count": len(decoded_lines),
            "paragraph_count": len(paragraphs) if paragraphs else (1 if paragraph_text else 0),
            "image_url": image_url,
            "preprocessed_image_url": processed_image_url,
        }
    ), 200


def _normalize_output_level(raw_level: str | None) -> str:
    allowed = {"all", "paragraph", "line", "word"}
    level = (raw_level or "all").strip().lower()
    return level if level in allowed else "all"


def _group_lines_into_paragraphs(
    lines: List[str],
    line_boxes: List[Tuple[int, int]],
) -> List[str]:
    if not lines:
        return []

    if len(lines) == 1:
        return [lines[0]]

    # Fallback when geometry is missing/misaligned.
    if len(line_boxes) != len(lines):
        merged = " ".join([line for line in lines if line]).strip()
        return [merged] if merged else []

    heights = [max(1, bottom - top) for top, bottom in line_boxes]
    median_height = float(np.median(np.array(heights, dtype=np.float32)))
    gap_threshold = max(14.0, 1.6 * median_height)

    paragraphs: List[str] = []
    current_paragraph_lines: List[str] = [lines[0]]

    for idx in range(1, len(lines)):
        prev_bottom = line_boxes[idx - 1][1]
        curr_top = line_boxes[idx][0]
        gap = float(curr_top - prev_bottom)

        if gap > gap_threshold:
            paragraph_text = " ".join(current_paragraph_lines).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)
            current_paragraph_lines = [lines[idx]]
        else:
            current_paragraph_lines.append(lines[idx])

    tail_text = " ".join(current_paragraph_lines).strip()
    if tail_text:
        paragraphs.append(tail_text)

    return paragraphs


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)