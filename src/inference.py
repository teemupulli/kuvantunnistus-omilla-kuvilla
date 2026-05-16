import os
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def get_class_names(train_dir):
    train_dir = Path(train_dir)
    if not train_dir.exists():
        return None
    classes = [p.name for p in sorted(train_dir.iterdir()) if p.is_dir()]
    return classes


def preprocess_image(path, image_size):
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size))
    arr = np.array(img).astype('float32') / 255.0
    return arr


def predict_image(model, img_array):
    x = np.expand_dims(img_array, axis=0)
    preds = model.predict(x)
    return preds[0]


def main():
    parser = argparse.ArgumentParser(description='Run inference with a saved Keras model')
    parser.add_argument('--model', default='aug_run_best.h5', help='Path to saved model (.h5 or .keras)')
    parser.add_argument('--input', required=True, help='Path to image file or folder with images')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--train_dir', default=os.path.join('..', 'dataset', 'split', 'train'), help='Path to train folder to infer class names')
    parser.add_argument('--out_csv', help='Optional CSV file to write results')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        return

    print('Loading model:', model_path)
    model = tf.keras.models.load_model(str(model_path))

    class_names = get_class_names(args.train_dir)
    if class_names:
        print('Class names from train dir:', class_names)
    else:
        # fallback: use numeric class indices
        class_names = None
        print('Train dir not found, will use numeric class indices')

    input_path = Path(args.input)
    if input_path.is_dir():
        files = [p for p in input_path.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        files = [input_path]

    results = []
    for f in files:
        try:
            img = preprocess_image(f, args.image_size)
            probs = predict_image(model, img)
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            label = class_names[top_idx] if class_names and top_idx < len(class_names) else str(top_idx)
            print(f.name, '->', label, f'({top_prob:.4f})')
            results.append((str(f), label, top_prob))
        except Exception as e:
            print('Error processing', f, e)

    if args.out_csv and results:
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['path', 'predicted_label', 'probability'])
            for row in results:
                writer.writerow(row)
        print('Results written to', args.out_csv)


if __name__ == '__main__':
    main()
