from spatialdata import read_zarr
from napari_spatialdata import Interactive

import napari

# Start Napari
viewer = napari.Viewer()
# sdata = read_zarr(r"C:\Users\matti\Documents\WERK\STAGE\VIB\data\ilastik_example_data\example.zarr")
# Interactive(sdata)
napari.run()
