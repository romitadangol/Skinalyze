print("Skinalyze prediction script started!")
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Locate the project root and saved model files
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "skinalyze_mobilenet_best.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

IMG_SIZE = (224, 224)


def load_skinalyze():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


def predict_skin_concern(image_path):
    model, class_names = load_skinalyze()

    # Load and resize the image in RGB format
    image = load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing is already inside the model
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
    }


if __name__ == "__main__":
    image_path = input(
        "Enter the path to a skin image: "
    ).strip().strip('"')

    try:
        result = predict_skin_concern(image_path)
        print("\nSkinalyze prediction")
        print("Skin concern:", result["prediction"])
        print("Model confidence:", f'{result["confidence"]}%')

    except (FileNotFoundError, ValueError) as error:
        print("Error:", error)
