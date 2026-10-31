import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Skin v2"
MODEL_PATH = PROJECT_DIR / "models" / "skinalyze_mobilenet_best.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
REPORT_PATH = PROJECT_DIR / "reports" / "figures" / "evaluation_report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = val_ds.class_names

model = tf.keras.models.load_model(MODEL_PATH)

images_list = []
labels_list = []

for images, labels in val_ds:
    images_list.append(images)
    labels_list.append(labels)

images = np.concatenate(images_list)
labels = np.concatenate(labels_list)

predictions = model.predict(images, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels, axis=1)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4
)

matrix = confusion_matrix(true_classes, predicted_classes)

output = (
    "SkinAlyze MobileNetV2 Evaluation\n"
    "--------------------------------\n"
    f"Classes: {class_names}\n\n"
    f"{report}\n\n"
    "Confusion Matrix:\n"
    f"{matrix}\n"
)

print(output)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(output)

print(f"Report saved to: {REPORT_PATH}")