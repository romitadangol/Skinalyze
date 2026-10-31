
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Skin v2"
MODEL_PATH = PROJECT_DIR / "models" / "skinalyze_mobilenet_best.keras"

OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

class_names = val_ds.class_names

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Collect predictions and actual labels
true_classes = []
predicted_classes = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)

    true_classes.extend(
        np.argmax(labels.numpy(), axis=1)
    )
    predicted_classes.extend(
        np.argmax(predictions, axis=1)
    )

true_classes = np.array(true_classes)
predicted_classes = np.array(predicted_classes)

# Calculate evaluation metrics
accuracy = accuracy_score(
    true_classes, predicted_classes
)

report = classification_report(
    true_classes,
    predicted_classes,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
)

matrix = confusion_matrix(
    true_classes,
    predicted_classes,
    labels=list(range(len(class_names)))
)

# Save text evaluation report
output = (
    "SkinAlyze MobileNetV2 Evaluation\n"
    "--------------------------------\n"
    f"Classes: {class_names}\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    f"{report}\n\n"
    "Confusion Matrix:\n"
    f"{matrix}\n"
)

print(output)

(OUTPUT_DIR / "evaluation_report.txt").write_text(
    output,
    encoding="utf-8"
)

# Generate confusion matrix image
fig, ax = plt.subplots(figsize=(9, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title("SkinAlyze - Confusion Matrix")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Generate classification report image
fig, ax = plt.subplots(figsize=(10, 5))

ax.axis("off")

ax.text(
    0.02,
    0.95,
    report,
    fontsize=11,
    family="monospace",
    va="top",
    transform=ax.transAxes
)

ax.set_title(
    "SkinAlyze - Classification Report",
    fontsize=15,
    pad=20
)

plt.savefig(
    OUTPUT_DIR / "classification_report.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Evaluation completed successfully!")
print(f"Images saved to: {OUTPUT_DIR}")
