import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename

from src.architecture import compile_hybrid_network
from src.preprocessing import execute_morphological_preprocessing
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

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    processed_tensor = execute_morphological_preprocessing(filepath)
    if processed_tensor is None:
        return jsonify({"ok": False, "error": "Preprocessing failed"}), 500

    # Create a visualization of the preprocessed tensor and save it to static/uploads
    try:
        # processed_tensor shape is (W, H, 1) after transpose in preprocessing
        vis_matrix = np.transpose(processed_tensor[..., 0])  # back to (H, W)
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

    image_url = url_for("static", filename=f"uploads/{filename}")
    return jsonify({"ok": True, "prediction": decoded[0], "image_url": image_url, "preprocessed_image_url": processed_image_url}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)