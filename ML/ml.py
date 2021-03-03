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
features = ["longitude", "latitude", "peak load"]

case_name = "ANBRIMEX"
df = pd.read_csv("results/" + case_name + "_final.csv", sep=";|,", engine="python", index_col='index')
#df = df.loc[df['off-grid'] == 1]
X = df[["longitude", "latitude", "peak load", "off-grid"]]
scaler.fit(X)
X = scaler.transform(X)
# X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

y = df["PV"]
cv = StratifiedKFold(12)
param_range = np.arange(1, 60, 1)
cv = KFold(n_splits=12, random_state=42, shuffle=True)

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
