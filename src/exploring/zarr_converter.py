from dask_image.imread import imread
import os
import random

from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

CWD = os.getcwd()
directory = r"D:\PPP_M17_SPC-035-full2\2024_09_20_M17_SPC-035_HelenaAegerter\R01\A01\ROI3"
name = "_".join(directory.split("\\")[-4:])
train_output = fr"D:\train\{name}.zarr"
test_output = fr"D:\test\{name}.zarr"
annotations_name = "annotations.tif"
train_test_split = .7

def main():
    print("STARTING CONVERTER")
    images = []

    for i in os.listdir(directory):
        if not i.startswith("C-000") and "DAPI" not in i and i.endswith(".tif") and i!=annotations_name:
            images.append(i)

    mask = imread(os.path.join(directory, annotations_name))

    random.shuffle(images)
    index = int(len(images)*train_test_split)

    train_list = images[:index]
    test_list = images[index:]

    train_sdata = SpatialData()
    test_sdata = SpatialData()


    print("PREPING DATA")
    for i in train_list:
        print(i)
        array = imread(os.path.join(directory,i))
        train_sdata[check_name(i)] = Image2DModel.parse(
        array,
        )

    for i in test_list:
        array = imread(os.path.join(directory,i))
        test_sdata[check_name(i)] = Image2DModel.parse(
        array,
        )

    train_sdata['annotations'] = Labels2DModel.parse(
        mask.squeeze(),
    )
    
    test_sdata['annitations'] = Labels2DModel.parse(
        mask.squeeze(),
    )

    print("SAVING DATA")
    train_sdata.write(
        train_output,
    )

    test_sdata.write(
        test_output,
    )

def check_name(string):
    checks = ["(", ")"]
    for c in checks:
        string = "_".join(string.split(c))
    
    return string

main()
