"""
CIFAKE - Flask Backend
AI-Generated Image Detection
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.h5"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.txt"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (32, 32)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp"
}


# Default class names
_class_names = ["FAKE", "REAL"]

_model = None


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    global _model
    global _class_names

    if _model is None:

        import tensorflow as tf

        print("Loading CIFAKE model...")
        print("Model path:", MODEL_PATH)

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                "Model file not found at: " + MODEL_PATH
            )

        try:

            _model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            print("Model loaded successfully.")

        except Exception as e:

            print("MODEL LOADING ERROR:", str(e))

            raise RuntimeError(
                "Could not load the TensorFlow model: " + str(e)
            )


        # Load class names if available

        if os.path.exists(CLASS_NAMES_PATH):

            try:

                with open(
                    CLASS_NAMES_PATH,
                    "r",
                    encoding="utf-8"
                ) as f:

                    content = f.read().strip()

                    if content:

                        _class_names = [
                            name.strip()
                            for name in content.split(",")
                            if name.strip()
                        ]

                print(
                    "Class names:",
                    _class_names
                )

            except Exception as e:

                print(
                    "Class names file error:",
                    str(e)
                )

    return _model


# ============================================================
# CHECK FILE EXTENSION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(file_bytes):

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Keep this consistent with the training process.
    # If your model was trained using /255.0,
    # change this to:
    #
    # image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("\n==============================")
    print("Prediction request received")
    print("==============================")

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        print("ERROR: image field missing")

        return jsonify({
            "error": "No image uploaded."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        print("ERROR: empty filename")

        return jsonify({
            "error": "No file selected."
        }), 400


    if not allowed_file(file.filename):

        print(
            "ERROR: unsupported file:",
            file.filename
        )

        return jsonify({
            "error": "Unsupported file type."
        }), 400


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = get_model()

    except FileNotFoundError as e:

        print(
            "MODEL NOT FOUND:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 503


    except Exception as e:

        print(
            "MODEL ERROR:",
            str(e)
        )

        return jsonify({
            "error": (
                "Model could not be loaded. "
                + str(e)
            )
        }), 500


    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    try:

        file_bytes = file.read()

        if not file_bytes:

            return jsonify({
                "error": "Uploaded image is empty."
            }), 400


        arr = preprocess_image(
            file_bytes
        )


        print(
            "Image shape:",
            arr.shape
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            arr,
            verbose=0
        )


        print(
            "Raw model output:",
            prediction
        )


        prob_real = float(
            prediction[0][0]
        )


    except Exception as e:

        print(
            "IMAGE/PREDICTION ERROR:",
            str(e)
        )

        return jsonify({
            "error": (
                "Failed to process image: "
                + str(e)
            )
        }), 500


    # --------------------------------------------------------
    # Convert probability to label
    # --------------------------------------------------------

    is_real = prob_real >= 0.5


    if is_real:

        label = (
            _class_names[1]
            if len(_class_names) > 1
            else "REAL"
        )

        confidence = prob_real

    else:

        label = (
            _class_names[0]
            if len(_class_names) > 0
            else "FAKE"
        )

        confidence = 1 - prob_real


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = {

        "label": label,

        "is_real": is_real,

        "confidence": round(
            confidence * 100,
            2
        ),

        "prob_real": round(
            prob_real * 100,
            2
        )

    }


    print(
        "Prediction result:",
        response
    )

    print("==============================\n")


    return jsonify(response)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    model_exists = os.path.exists(
        MODEL_PATH
    )

    return jsonify({

        "status": "ok",

        "model_ready": model_exists,

        "model_path": MODEL_PATH

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
