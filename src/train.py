import argparse

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_pipeline import create_generators, split_dataset
from models import build_simple_cnn, build_feature_extraction_model, build_fine_tuned_model


def parse_args():
    parser = argparse.ArgumentParser(description="Kuvantunnistusmallien koulutus")
    parser.add_argument("--mode", choices=["cnn", "feature", "finetune"], default="cnn")
    parser.add_argument("--data_dir", default="dataset/split")
    parser.add_argument("--raw_dir", default="dataset/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--unfrozen_layers", type=int, default=4)
    return parser.parse_args()


def build_model(mode, input_shape, num_classes, unfrozen_layers):
    if mode == "cnn":
        return build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    if mode == "feature":
        return build_feature_extraction_model(input_shape=input_shape, num_classes=num_classes)
    if mode == "finetune":
        return build_fine_tuned_model(input_shape=input_shape, num_classes=num_classes, unfrozen_layers=unfrozen_layers)
    raise ValueError(f"Tuntematon mode: {mode}")


def main():
    args = parse_args()
    split_dataset(args.raw_dir, args.data_dir)

    train_gen, val_gen, test_gen = create_generators(
        args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    num_classes = len(train_gen.class_indices)
    input_shape = (*train_gen.image_shape,)
    model = build_model(args.mode, input_shape=input_shape, num_classes=num_classes, unfrozen_layers=args.unfrozen_layers)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("model_best.h5", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Evaluating on test data...")
    results = model.evaluate(test_gen)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])


if __name__ == "__main__":
    main()
