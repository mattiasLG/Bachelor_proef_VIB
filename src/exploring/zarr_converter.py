from dask_image.imread import imread
import os
import random

from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

CWD = os.getcwd()
file = r"C:\Users\Mattias\Documents\projects\HOGENT\Bach_proef\Bachelor_proef_VIB\src\model\files.txt"
annotations_name = "annotations.tif"
train_test_split = .8
overwrite = True

def main():
    print("STARTING CONVERTER")
    directories = open(file, "r")

    for dir in directories.read().split("\n"):
        images = []
        for i in os.listdir(dir):
            if not i.startswith("C-000") and "DAPI" not in i and i.endswith(".tif") and i!=annotations_name:
                images.append(i)

        mask = imread(os.path.join(dir, annotations_name))

        random.shuffle(images)
        index = int(len(images)*train_test_split)

        train_list = images[:index]
        test_list = images[index:]

        train_sdata = SpatialData()
        test_sdata = SpatialData()

        print("PREPING DATA")
        for i in train_list:
            array = imread(os.path.join(dir,i))
            train_sdata[check_name(i)] = Image2DModel.parse(
            array,
            chunks=(512, 512)
            )

        for i in test_list:
            array = imread(os.path.join(dir,i))
            test_sdata[check_name(i)] = Image2DModel.parse(
            array,
            chunks=(512, 512)
            )

        train_sdata['annotations'] = Labels2DModel.parse(
            mask.squeeze(),
        )
        
        test_sdata['annotations'] = Labels2DModel.parse(
            mask.squeeze(),
        )

        print("SAVING DATA")
        name = "_".join(dir.split("\\")[-4:])
        train_output = fr"E:\train\{name}.zarr"
        test_output = fr"E:\test\{name}.zarr"

        print(name)
        train_sdata.write(
            train_output,
            overwrite,
        )

        test_sdata.write(
            test_output,
            overwrite,
        )

def check_name(string):
    checks = ["(", ")"]
    for c in checks:
        string = "_".join(string.split(c))
    
    return string

main()
