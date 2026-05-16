import os
import argparse
import json
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_pipeline import create_generators
from models import build_simple_cnn, build_feature_extraction_model, build_fine_tuned_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cnn', 'feature', 'finetune'], default='cnn')
    parser.add_argument('--data_dir', default=os.path.join('..', 'dataset', 'split'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--unfrozen_layers', type=int, default=4)
    parser.add_argument('--strong_augment', action='store_true')
    parser.add_argument('--save_prefix', default='run')
    return parser.parse_args()


def save_history(history, out_path):
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out_path, 'w') as f:
        json.dump(hist, f)


def plot_history(history, out_prefix):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'b-', label='Train acc')
    plt.plot(epochs, val_acc, 'r--', label='Val acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f'{out_prefix}_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'b-', label='Train loss')
    plt.plot(epochs, val_loss, 'r--', label='Val loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f'{out_prefix}_loss.png')
    plt.close()


def compute_class_weights(generator):
    from collections import Counter
    classes = generator.classes
    counts = Counter(classes.tolist())
    total = sum(counts.values())
    class_weight = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return class_weight


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    print('Using data_dir:', data_dir)

    train_gen, val_gen, test_gen = create_generators(data_dir, image_size=(args.image_size, args.image_size), batch_size=args.batch_size)

    num_classes = len(train_gen.class_indices)
    input_shape = (*train_gen.image_shape,)
    print('Classes:', train_gen.class_indices)

    if args.mode == 'cnn':
        model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    elif args.mode == 'feature':
        model = build_feature_extraction_model(input_shape=input_shape, num_classes=num_classes)
    else:
        model = build_fine_tuned_model(input_shape=input_shape, num_classes=num_classes, unfrozen_layers=args.unfrozen_layers)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    class_weight = compute_class_weights(train_gen)
    print('Class weights:', class_weight)

    out_prefix = args.save_prefix
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True), ModelCheckpoint(f'{out_prefix}_best.h5', save_best_only=True)]

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks, class_weight=class_weight)

    # Save history and plots
    hist_path = f'{out_prefix}_history.json'
    save_history(history, hist_path)
    plot_history(history, out_prefix)

    print('Evaluating on test set...')
    res = model.evaluate(test_gen)
    print('Test loss, Test acc:', res)


if __name__ == '__main__':
    main()
