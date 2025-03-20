from dask_image.imread import imread
import os

from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

DIRECTORY = r"C:\Users\matti\Documents\HOGENT\Jaar-3\bachproef\Bachelor_proef_VIB\Testdata_MACSima\Testdata_CE\R01\B01\ROI1"
ANNOTATION = r"C:\Users\matti\Documents\HOGENT\Jaar-3\bachproef\Bachelor_proef_VIB\Testdata_MACSima\Testdata_CE\dummy_acquisition_bleaching.tif"
OUTPUT = r"C:\Users\matti\Documents\HOGENT\Jaar-3\bachproef\Bachelor_proef_VIB\Testdata_MACSima\Testdata_CE.zarr"

def main():
    images = []

    for file in os.listdir(DIRECTORY):
        array = imread(os.path.join(DIRECTORY,file))
        images.append(array)

    mask = imread(ANNOTATION)

    mask

    sdata = SpatialData()

    for i in range(len(images)):
        print(i)
        sdata[f"channel_{i}"] = Image2DModel.parse(
        images[i],
    )

    print(mask)
    sdata['labels'] = Labels2DModel.parse(
        mask.squeeze(),
    )
    sdata

    sdata.write(
        OUTPUT,
        overwrite=True,
    )

main()
