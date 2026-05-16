import os
import argparse
from collections import Counter

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_pipeline import create_generators
from models import build_simple_cnn


def compute_class_weights(generator):
    # generator.classes is a numpy array of class indices
    classes = generator.classes
    counts = Counter(classes.tolist())
    total = sum(counts.values())
    class_weight = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return class_weight


def parse_args():
    parser = argparse.ArgumentParser(description='Quick training script')
    parser.add_argument('--data_dir', default=os.path.join('..', 'dataset', 'split'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    print('Using data_dir:', data_dir)

    train_gen, val_gen, test_gen = create_generators(data_dir, image_size=(args.image_size, args.image_size), batch_size=args.batch_size)

    num_classes = len(train_gen.class_indices)
    input_shape = (*train_gen.image_shape,)
    print('Classes:', train_gen.class_indices)
    print('Num classes:', num_classes)

    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    class_weight = compute_class_weights(train_gen)
    print('Computed class weights:', class_weight)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('quick_model_best.h5', save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print('Evaluating on test set...')
    res = model.evaluate(test_gen)
    print('Test loss, Test acc:', res)


if __name__ == '__main__':
    main()
