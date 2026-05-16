import os
import random
import shutil
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    random.seed(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    for subset in ["train", "val", "test"]:
        ensure_dir(target_dir / subset)

    classes = [item.name for item in source_dir.iterdir() if item.is_dir()]
    classes.sort()

    for class_name in classes:
        class_source = source_dir / class_name
        images = [p for p in class_source.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not images:
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for subset, subset_images in splits.items():
            target_class = target_dir / subset / class_name
            ensure_dir(target_class)
            for img_path in subset_images:
                dest_path = target_class / img_path.name
                shutil.copy2(img_path, dest_path)

    return target_dir


def create_generators(data_dir, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        channel_shift_range=20.0,
        fill_mode="nearest",
    )

    test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator
