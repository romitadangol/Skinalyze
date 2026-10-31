import base64
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.utils import img_to_array


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "skinalyze_mobilenet_best.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
PRODUCTS_PATH = PROJECT_DIR / "products.json"

IMG_SIZE = (224, 224)

CATEGORIES = [
    "Cleanser",
    "Toner",
    "Serum",
    "Moisturizer",
    "Sunscreen",
]


INGREDIENT_NAMES = {
    "acne": ["Salicylic acid", "Azelaic acid"],
    "blackheads": ["Salicylic acid", "Glycolic acid"],
    "dark spots": ["Azelaic acid", "Vitamin C", "Alpha arbutin"], 
    "pores": ["Salicylic acid", "Niacinamide"],
    "wrinkles": ["Retinol", "Hyaluronic acid", "Bakuchiol"]
}

RETAILER_HOSTS = {
    "koreanbeautypoint.com",
    "www.koreanbeautypoint.com",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with CLASS_NAMES_PATH.open(encoding="utf-8") as file:
    class_names = json.load(file)

if model.output_shape[-1] != len(class_names):
    raise ValueError("Model output does not match class_names.json.")


def normalize(value):
    return str(value or "").strip().lower()


def safe_product_url(value):
    value = str(value or "").strip()
    parsed = urlparse(value)

    if (
        parsed.scheme == "https"
        and parsed.hostname in RETAILER_HOSTS
        and parsed.username is None
        and parsed.password is None
    ):
        return value

    return ""


def safe_image_url(value):
    value = str(value or "").strip()

    if value.startswith("/static/"):
        return value

    parsed = urlparse(value)

    if parsed.scheme == "https" and parsed.hostname:
        return value

    return ""


def build_routine(concern):
    with PRODUCTS_PATH.open(encoding="utf-8") as file:
        catalogue = json.load(file)

    if not isinstance(catalogue, list):
        raise ValueError("products.json must contain a JSON list.")

    routine = []

    for category in CATEGORIES:
        matches = []

        for item in catalogue:
            if not isinstance(item, dict):
                continue

            if normalize(item.get("category")) != normalize(category):
                continue

            concerns = item.get("concerns", [])

            if not isinstance(concerns, list):
                raise ValueError(
                    "Each product's concerns field must be a list."
                )

            if concern not in {normalize(value) for value in concerns}:
                continue

            if not str(item.get("name", "")).strip():
                continue

            matches.append({
                "name": item["name"],
                "description": item.get("description", ""),
                "url": safe_product_url(item.get("url")),
                "image": safe_image_url(item.get("image")),
            })

        routine.append({
            "category": category,
            "products": matches,
        })

    return routine


def predict_image(image):
    resized = image.resize(IMG_SIZE, Image.Resampling.NEAREST)
    array = img_to_array(resized)
    array = np.expand_dims(array, axis=0)

    # Preserve the existing pipeline.
    # MobileNetV2 preprocessing is inside the saved model.
    probabilities = model.predict(array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    concern = normalize(class_names[predicted_index])

    return {
        "concern": concern,
        "prediction": concern.title(),
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
    }


def create_preview(image):
    preview = image.copy()
    preview.thumbnail((640, 640))

    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def show_page(**values):
    context = {
        "result": None,
        "preview": None,
        "ingredients": [],
        "routine": [],
        "error": None,
    }

    context.update(values)
    return render_template("index.html", **context)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return show_page()

    upload = request.files.get("image")

    if not upload or not upload.filename:
        return show_page(
            error="Please choose an image first."
        ), 400

    try:
        with Image.open(upload.stream) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                return show_page(
                    error="Please upload a JPG, PNG or WebP image."
                ), 400

            if source.width * source.height > 20_000_000:
                return show_page(
                    error="Please use an image smaller than 20 megapixels."
                ), 400

            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

        result = predict_image(image)

        return show_page(
            result=result,
            preview=create_preview(image),
            ingredients=INGREDIENT_NAMES.get(result["concern"], []),
            routine=build_routine(result["concern"]),
        )

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ):
        return show_page(
            error="The image could not be read. Try another image."
        ), 400

    except Exception:
        app.logger.exception("SkinAlyze analysis failed")

        return show_page(
            error=(
                "Analysis could not be completed. "
                "Check the Terminal for error details."
            )
        ), 500


@app.errorhandler(413)
def too_large(error):
    return show_page(
        error="The upload is too large. Choose an image under 8 MB."
    ), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )