import graphviz
import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


print(os.getcwd())
data = pd.read_csv('../Data/WineDataset/wine.data.csv' )
X = data.values[:, :-1]
y = data.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
dot_data_xg = xgb.to_graphviz(xgb_clf)
graph = graphviz.Source(dot_data_xg)
# graph.render("xgb_tree_wine")
y_hat = xgb_clf.predict(X_test)
plot_confusion_matrix(xgb_clf, X_test, y_test, display_labels=['class 0', 'class 1', 'class 2'], values_format='.0f')
X_test_leaves = xgb_clf.apply(X_test)
# plt.show(block=True)
x = 0
