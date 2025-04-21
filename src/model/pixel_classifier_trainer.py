import sys
import os
import numpy as np
import logging
import keras

import gc
from tensorflow.keras import backend as K

from spatialdata import SpatialData
from spatialdata import read_zarr
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from skimage.util import view_as_windows

CWD = os.getcwd()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_SIZE = 256
STRIDE = IMAGE_SIZE - 32
MODEL_NAME = os.path.join(FILE_PATH, f"pixel_classifier_{IMAGE_SIZE}.keras")
TO_SAVE = True
BATCH_SIZE = 2
EPOCHS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console
        logging.FileHandler(os.path.join(FILE_PATH, "training.log"))  # File
    ]
)

logger = logging.getLogger("Pixel Classifier")

def main(dir):
    print("running")

    if os.path.isfile(MODEL_NAME):
            logger.info("Fetching model")
            model = keras.models.load_model(MODEL_NAME)
    else:
        logger.info("making model")
        model = unet_model_smaller()

    for arg in os.listdir(dir):
        path = os.path.join(dir, arg)
        if os.path.isdir(path) and arg.endswith(".zarr"):
            sdata = read_zarr(path)
            logger.info(f"training on {path} dataset")
            workflow(sdata, model)
        else:
            logger.warning(f"{path} is no zarr file. Please pass one.")

def workflow(sdata:SpatialData, model):
    logger.info("fetching data")
    keys = list(sdata.images.keys())
    # labels = extract_patches(sdata['annotations'].data)//255
    labels = extract_patches_skimage(sdata['annotations'].values)//255

    for i in keys[:1]:
        # data = extract_patches(sdata[i].data.squeeze())/255
        data = extract_patches_skimage(sdata[i].values.squeeze())/255

        X = data[..., np.newaxis]  # shape: (num_patches, 256, 256, 1)
        Y = labels[..., np.newaxis]

        train_model(model, X, Y)

        K.clear_session()
        gc.collect()

        if TO_SAVE:
            logger.info("SAVING MODEL")
            model.save(MODEL_NAME)

def train_model(model, X, Y):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("TRAINING MODEL")
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    

def unet_model_smaller(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):

    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

def extract_patches(array:np.ndarray):
    patches = []
    h, w = array.shape

    # Ensure that the image dimensions are divisible by the patch size and stride
    for i in range(0, h - IMAGE_SIZE + 1, STRIDE):
        for j in range(0, w - IMAGE_SIZE + 1, STRIDE):
            img_patch = array[i:i+IMAGE_SIZE, j:j+IMAGE_SIZE]
            patches.append(img_patch)

    return np.array(patches)

def extract_patches_skimage(array: np.ndarray):
    window_shape = (IMAGE_SIZE, IMAGE_SIZE)
    step = STRIDE
    patches = view_as_windows(array, window_shape, step=step)
    patches = patches.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    return patches



main(sys.argv[1])