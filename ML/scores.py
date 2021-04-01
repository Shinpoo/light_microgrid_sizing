import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression


scaler = StandardScaler()
# neigh = KNeighborsRegressor(n_neighbors=5)
# regression_visualizers = [ResidualsPlot(neigh), PredictionError(neigh)]
features = ["longitude", "latitude", "peak_load", "off-grid","avg_peak_winter","avg_peak_spring","avg_peak_summer","avg_peak_autumn","avg_base_winter","avg_base_spring","avg_base_summer","avg_base_autumn"]

case_name = "mg_sizing_dataset_with_loc"
df = pd.read_csv("results/" + case_name + ".csv", sep=";|,", engine="python", index_col='index')
X = df[features]
scaler.fit(X)
X = scaler.transform(X)
targets = ["PV","BAT","RBAT","INV","GEN","NPV"]
y = df[targets[0]]

model = LinearRegression()
visualizer_residuals = ResidualsPlot(model)
visualizer_residuals.fit(X, y)
visualizer_residuals.show()