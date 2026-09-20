"""
app.py — Flask backend for the CIFAKE image classifier.

Serves the frontend (templates/index.html) and a /predict API endpoint that
takes an uploaded image, runs it through the trained model, and returns
whether it looks REAL or AI-GENERATED (FAKE) with a confidence score.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.h5")
CLASS_NAMES_PATH = os.path.join("model", "class_names.txt")
IMG_SIZE = (32, 32)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

_model = None
_class_names = ["FAKE", "REAL"]  # default alphabetical order, overwritten if file exists


def get_model():
    """Lazy-load the Keras model so the server starts even before training."""
    global _model
    if _model is None:
        import tensorflow as tf  # imported here to keep server startup fast

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. Run train.py first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH) as f:
                global _class_names
                _class_names = f.read().strip().split(",")
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    image_array = np.array(img, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    try:
        arr = preprocess_image(file.read())
        prob_real = float(model.predict(arr, verbose=0)[0][0])
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    # class_names is alphabetical: index 0 = FAKE, index 1 = REAL (sigmoid output = P(REAL))
    is_real = prob_real >= 0.5
    confidence = prob_real if is_real else (1 - prob_real)
    label = _class_names[1] if is_real else _class_names[0]

    return jsonify({
        "label": label,
        "is_real": is_real,
        "confidence": round(confidence * 100, 2),
        "prob_real": round(prob_real * 100, 2),
    })


@app.route("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({"status": "ok", "model_ready": model_ready})


if __name__ == "__main__":
    app.run(debug=True)
