from osgeo import gdal, ogr
import os
import argparse

DIR_PATH = os.path.dirname(__file__)

geojson_path = r"D:\2024_03_15_CelineEveraert_3_240414_071301_preprocessed\2024_03_15_CelineEveraert_3\R01\B01\ROI1\annotations.geojson"
reference_tiff = r"D:\2024_03_15_CelineEveraert_3_240414_071301_preprocessed\2024_03_15_CelineEveraert_3\R01\B01\ROI1\C-001_S-000_B_FITC_R-01_W-B01_ROI-01_A-Bcl_2_C-REA872.tif"
output_tiff = geojson_path.split(".")[0]+".tif"

src_ds = gdal.Open(reference_tiff)
geo_transform = src_ds.GetGeoTransform()
projection = src_ds.GetProjection()
x_res = src_ds.RasterXSize
y_res = src_ds.RasterYSize

vector_ds = ogr.Open(geojson_path)
layer = vector_ds.GetLayer()

target_ds = gdal.GetDriverByName('GTiff').Create(output_tiff, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(geo_transform)
target_ds.SetProjection(projection)

gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[255])

# Close datasets
target_ds = None
vector_ds = None
src_ds = None

print(f"Annotations rasterized to {output_tiff}, matching {reference_tiff}")
