import sys
import os
import joblib
import dask.array as da
import logger

from sklearn.ensemble import RandomForestClassifier
from spatialdata import read_zarr

MODEL_NAME = "ML_MODEL.pkl"

def main(file):
    print("running")

    sdata = read_zarr(file)

    model = Custom_Random_Forest(n_estimators=50, n_jobs=5, max_depth=10)

    X = []
    y = []

    for i in range(len(sdata.labels)):
        X.append(sdata[f"channel_{i}"])
        y.append(sdata[f"label_{i}"])

    model = Custom_Random_Forest(n_estimators=50, n_jobs=5, max_depth=10)

    model.fit(X, y)

class Custom_Random_Forest(RandomForestClassifier):
    def predict(self, X):
        images, feutures, H, W = X.shape
        X = da.transpose(X, (0, 2, 3, 1))

        X = X.reshape(images*H * W, feutures)
        pred = super().predict(X)
        return pred.reshape(images, H, W)
     

main(sys.argv[1])