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
MODEL_NAME = f"pixel_classifier_{IMAGE_SIZE}"
TO_SAVE = True
BATCH_SIZE = 50
EPOCHS = 5

logger = logging.getLogger("Pixel Classifier")

def main(args):
    for arg in args:
        path = os.path.join(CWD, arg)
        if os.path.isdir(path) and arg.endswith(".zarr"):
            sdata = read_zarr(path)

            workflow(sdata)
        else:
            logger.warning(f"{path} is no zarr file. Please pass one.")

def workflow(sdata:SpatialData):
    keys = list(sdata.images.keys())

    for i in keys[:1]:
        data = image_splitter(sdata[i].data.squeeze())
        labels = image_splitter(sdata['labels'].data.squeeze())

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

def image_stitcher(data:np.ndarray, shape, image_slice_size=IMAGE_SIZE):
    x = shape[0]//image_slice_size
    y = shape[1]//image_slice_size

    vert = []
    print(x, y)

    for i in range(x):
        print(data[i*y:i*y+y])
        vert.append(np.concatenate(data[i*y:i*y+y], axis=1))

    return np.concatenate(vert, axis=0)

def image_splitter(array:np.ndarray, image_slice_size=IMAGE_SIZE):
    shape = array.shape
    images = []
    for i in range(shape[0]//image_slice_size):
        for j in range(shape[1]//image_slice_size):
            images.append(array[i*image_slice_size:i*image_slice_size+image_slice_size, j*image_slice_size:j*image_slice_size+image_slice_size])

    return np.array(images)

main(sys.argv[1:])