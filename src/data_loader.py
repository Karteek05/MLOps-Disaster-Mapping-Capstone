import os
import sys
import glob
import numpy as np
import tensorflow as tf


def _collapse_mask_channels(mask, num_classes):
    c = tf.shape(mask)[-1]

    def squeeze_chan():
        return tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    def argmax_chan():
        return tf.cast(tf.argmax(mask, axis=-1, output_type=tf.int32), tf.int32)

    return tf.case(
        [(tf.equal(c, 1), squeeze_chan), (tf.equal(c, num_classes), argmax_chan)],
        default=argmax_chan,
        exclusive=True,
    )


def make_load_and_parse(img_size, num_channels, num_classes):
    """Returns a load_and_parse(image_path, mask_path) fn bound to these dims."""

    def load_and_parse(image_path_tensor, mask_path_tensor):
        def _load_npy(path):
            return np.load(path.decode()).astype(np.float32)

        image = tf.numpy_function(_load_npy, [image_path_tensor], tf.float32)
        image.set_shape([None, None, num_channels])
        image = tf.image.resize(image, [img_size, img_size])

        mask_bytes = tf.io.read_file(mask_path_tensor)
        mask = tf.io.decode_image(mask_bytes, channels=0, dtype=tf.uint8)
        if mask.shape.rank is None:
            mask.set_shape([None, None, None])

        mask = tf.cond(
            tf.equal(tf.rank(mask), 3),
            lambda: _collapse_mask_channels(mask, num_classes),
            lambda: tf.cast(mask, tf.int32),
        )
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(tf.cast(mask, tf.float32), [img_size, img_size], method="nearest")
        mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)
        mask.set_shape([img_size, img_size])

        return image, mask

    return load_and_parse


def list_train_val_files(data_dir, val_split=0.2, seed=42):
    """Same file listing + shuffle + split used for both training and evaluation,
    so evaluate.py scores the model on the exact held-out set it never trained on."""
    all_mask_paths = sorted(glob.glob(os.path.join(data_dir, "*_mask.png")))
    if not all_mask_paths:
        print(f"FATAL: No mask files found in {data_dir}. Run `dvc pull` / `dvc repro prepare` first.")
        sys.exit(1)

    image_paths, mask_paths = [], []
    for mask_path in all_mask_paths:
        image_path = mask_path.replace("_mask.png", "_stacked.npy")
        if os.path.exists(image_path):
            image_paths.append(image_path)
            mask_paths.append(mask_path)

    dataset_size = len(image_paths)
    train_size = int(dataset_size * (1 - val_split))

    order = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    image_paths = np.array(image_paths)[order]
    mask_paths = np.array(mask_paths)[order]

    train_images, val_images = image_paths[:train_size], image_paths[train_size:]
    train_masks, val_masks = mask_paths[:train_size], mask_paths[train_size:]
    return (list(train_images), list(train_masks)), (list(val_images), list(val_masks))


def build_dataset(image_paths, mask_paths, img_size, num_channels, num_classes, batch_size):
    load_and_parse = make_load_and_parse(img_size, num_channels, num_classes)
    autotune = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_and_parse, num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size).prefetch(autotune)
    return dataset


def load_real_data(data_dir, img_size, num_channels, num_classes, batch_size, val_split=0.2, seed=42):
    (train_images, train_masks), (val_images, val_masks) = list_train_val_files(data_dir, val_split, seed)
    print(f"Data ready: {len(train_images)} train, {len(val_images)} val")
    train_dataset = build_dataset(train_images, train_masks, img_size, num_channels, num_classes, batch_size)
    val_dataset = build_dataset(val_images, val_masks, img_size, num_channels, num_classes, batch_size)
    return train_dataset, val_dataset
