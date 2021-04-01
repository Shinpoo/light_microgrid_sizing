import pandas as pd
from yellowbrick.features import RadViz, rank1d, rank2d, PCA, pca_decomposition, joint_plot, manifold_embedding
from sklearn.preprocessing import StandardScaler
from yellowbrick.target import balanced_binning_reference
from yellowbrick.target.feature_correlation import feature_correlation

import matplotlib.pyplot as plt

case_name = "mg_sizing_dataset_with_loc"
df = pd.read_csv("results/" + case_name + ".csv", sep=";|,", engine="python", index_col='index')

features = ["longitude", "latitude", "peak_load", "off-grid","avg_peak_winter","avg_peak_spring","avg_peak_summer","avg_peak_autumn","avg_base_winter","avg_base_spring","avg_base_summer","avg_base_autumn"]
target = ["PV","BAT"]

scaler = StandardScaler()
#df = df.loc[df['off-grid'] == 1]
X = df[features]
scaler.fit(X)
# X = scaler.transform(X)
X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
y = df[target]

PCA_ = False
if PCA_:
    features_visualizers = [PCA(scale=True, projection=3, proj_features=True, features=features)]
else:
    # features_visualizers = [Rank1D(algorithm='shapiro', features=features),
    #                         Rank2D(algorithm='pearson', features=features),
    #                         JointPlotVisualizer(features=features, columns=features[0]),
    #                         JointPlotVisualizer(features=features, columns=features[1]),
    #                         JointPlotVisualizer(features=features, columns=features[2]),
    #                         # Manifold(manifold="isomap", n_neighbors=10, features=features)
    #                         ]
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))

    rank1d(X, ax=axes[0, 0], show=False)
    rank2d(X, ax=axes[1, 0], show=False)
    pca_decomposition(X, y, scale=True, proj_features=True, ax=axes[0, 1], show=False)
    manifold_embedding(X, y, manifold="isomap", n_neighbors=10, ax=axes[1, 1])

    plt.show()

    visualizer = PCA(scale=True, proj_features=True, projection=3)
    visualizer.fit_transform(X, y)
    visualizer.show()

    for f in features:
        joint_plot(X, y, columns=f)

    balanced_binning_reference(y, bins=3)
    feature_correlation(X, y, labels=features)
