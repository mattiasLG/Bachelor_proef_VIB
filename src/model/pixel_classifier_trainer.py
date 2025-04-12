import sys
import os
import numpy as np
import logging
import keras

from spatialdata import SpatialData
from spatialdata import read_zarr
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model

CWD = os.getcwd()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_SIZE = 512
MODEL_NAME = os.path.join(FILE_PATH, f"pixel_classifier_{IMAGE_SIZE}.keras")
TO_SAVE = True
BATCH_SIZE = 8
EPOCHS = 10

logger = logging.getLogger("Pixel Classifier")

def main(args):
    print("running")
    for arg in args:
        path = os.path.join(CWD, arg)
        if os.path.isdir(path) and arg.endswith(".zarr"):
            sdata = read_zarr(path)

            workflow(sdata)
        else:
            logger.warning(f"{path} is no zarr file. Please pass one.")

def workflow(sdata:SpatialData):
    keys = list(sdata.images.keys())

    labels = extract_patches(sdata['labels'].data.squeeze()) // 255

    for i in keys:
        data = extract_patches(sdata[i].data.squeeze()) / 255

        if os.path.isdir(os.path.join(FILE_PATH, MODEL_NAME)):
            model = keras.models.load_model("saved_model_directory")
        else:
            model = unet_model_smaller()

        train_model(model, data, labels)


def train_model(model, X, Y):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("TRAINING MODEL")
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

    if TO_SAVE:
        logger.info("SAVING MODEL")
        model.save(MODEL_NAME)
    

def unet_model_smaller(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):

    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

def extract_patches(array:np.ndarray, patch_size=512, stride=64):
    patches = []
    h, w = array.shape

    for i in range(0, h - patch_size, stride):
        for j in range(0, w - patch_size, stride):
            img_patch = array[i:i+patch_size, j:j+patch_size]

            patches.append(img_patch)

    return np.array(patches)

main(sys.argv[1:])