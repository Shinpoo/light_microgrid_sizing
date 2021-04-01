import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores
from yellowbrick.regressor import ResidualsPlot, PredictionError

scaler = StandardScaler()
# neigh = KNeighborsRegressor(n_neighbors=5)
# regression_visualizers = [ResidualsPlot(neigh), PredictionError(neigh)]
features = ["longitude", "latitude", "peak_load", "off-grid","avg_peak_winter","avg_peak_spring","avg_peak_summer","avg_peak_autumn","avg_base_winter","avg_base_spring","avg_base_summer","avg_base_autumn"]

case_name = "mg_sizing_dataset_with_loc"
df = pd.read_csv("results/" + case_name + ".csv", sep=";|,", engine="python", index_col='index')
#df = df.loc[df['off-grid'] == 1]
X = df[features]
scaler.fit(X)
X = scaler.transform(X)
# X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
targets = ["PV","BAT","RBAT","INV","GEN","NPV"]
y = df[targets]
cv = StratifiedKFold(12)
param_range = np.arange(1, 30, 1)
cv = KFold(n_splits=12, random_state=40, shuffle=True)

viz = ValidationCurve(
    KNeighborsRegressor(), param_name="n_neighbors", param_range=param_range, scoring="r2", cv=cv, n_jobs=8
)

viz.fit(X, y)
viz.show()

visualizer = LearningCurve(KNeighborsRegressor(), scoring='r2', random_state=2, cv=cv, shuffle=True)

visualizer.fit(X, y)
visualizer.show()

vis = CVScores(KNeighborsRegressor(), cv=cv, scoring='r2')

vis.fit(X, y)        # Fit the data to the visualizer
vis.show()
