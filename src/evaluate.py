import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_test_generator(test_dir, image_size, batch_size=16):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def save_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Keras image classifier on test data")
    parser.add_argument("--model", required=True, help="Path to saved Keras model file")
    parser.add_argument("--test_dir", default="dataset/split/test", help="Test data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--image_size", type=int, default=224, help="Image width and height")
    parser.add_argument("--output_prefix", default="eval", help="Prefix for output files")
    args = parser.parse_args()

    test_generator = build_test_generator(args.test_dir, args.image_size, args.batch_size)
    class_names = [name for name, _ in sorted(test_generator.class_indices.items(), key=lambda item: item[1])]

    print("Loading model:", args.model)
    model = load_model(args.model)

    print("Running predictions on test data...")
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    report_json_path = f"{args.output_prefix}_classification_report.json"
    report_txt_path = f"{args.output_prefix}_classification_report.txt"
    cm_path = f"{args.output_prefix}_confusion_matrix.png"

    with open(report_json_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)

    with open(report_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(report_text)

    save_confusion_matrix(cm, class_names, cm_path)

    print("Evaluation results saved:")
    print(" -", report_json_path)
    print(" -", report_txt_path)
    print(" -", cm_path)
    print()
    print(report_text)


if __name__ == "__main__":
    main()
