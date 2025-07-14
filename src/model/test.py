import os
import sys
import numpy as np
import tensorflow as tf
from spatialdata import read_zarr
from tensorflow.keras import backend as K
from skimage.util import view_as_windows

# Constants
IMAGE_SIZE = 256
STRIDE = IMAGE_SIZE - 32
MODEL_NAME = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"pixel_classifier_{IMAGE_SIZE}.keras")

def main(test_dir):
    # Load model
    if not os.path.exists(MODEL_NAME):
        print(f"Model not found: {MODEL_NAME}")
        return
    
    model = tf.keras.models.load_model(
        MODEL_NAME,
        custom_objects={'bce_dice_loss': bce_dice_loss, 'binary_iou': binary_iou}
    )

    # Iterate over test zarr datasets
    for entry in os.listdir(test_dir):
        path = os.path.join(test_dir, entry)
        if os.path.isdir(path) and entry.endswith(".zarr"):
            print(f"Evaluating on {path}")
            sdata = read_zarr(path)
            evaluate_on_data(sdata, model)
        else:
            print(f"Skipping {entry}, not a .zarr directory.")

def evaluate_on_data(sdata, model):
    keys = list(sdata.images.keys())
    labels = sdata['annotations'].values // 255

    all_ious = []

    for key in keys[:1]:  # Limit to first 5 for sanity
        image = sdata[key].values.squeeze()
        image = min_max_scaler(image)

        image_patches = extract_patches_skimage(image)
        label_patches = extract_patches_skimage(labels)

        if image_patches.shape[0] != label_patches.shape[0]:
            print(f"Skipping {key}: Patch mismatch")
            continue

        X = image_patches[..., np.newaxis].astype(np.float16)
        Y = label_patches[..., np.newaxis].astype(np.float16)

        preds = model.predict(X, batch_size=2)
        iou = binary_iou_np(Y, preds)

        all_ious.append(iou)

        print(f"{key} — IoU: {iou:.4f}")

    print(f"\nAverage IoU: {np.mean(all_ious):.4f}")

def extract_patches_skimage(array):
    h, w = array.shape
    pad_h = (IMAGE_SIZE - h % IMAGE_SIZE) % IMAGE_SIZE
    pad_w = (IMAGE_SIZE - w % IMAGE_SIZE) % IMAGE_SIZE

    array = np.pad(array, ((pad_h//2, (pad_h - pad_h//2)), (pad_w//2, (pad_w - pad_w//2))), mode='constant')

    patches = view_as_windows(array, (IMAGE_SIZE, IMAGE_SIZE), step=STRIDE)
    return patches.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

def min_max_scaler(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-6)

# Metric functions
def binary_iou_np(y_true, y_pred, threshold=0.5):
    y_true = y_true.astype(np.float32)
    y_pred = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection + 1e-6) / (union + 1e-6)

# TensorFlow metric/loss for loading the model
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def binary_iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <test_data_dir>")
        sys.exit(1)

    main(sys.argv[1])
