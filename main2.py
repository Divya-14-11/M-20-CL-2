from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

x,y = make_classification(n_samples = 100, n_features = 1, n_classes = 2, n_clusters_per_class= 1, flip_y = 0.03, n_informative=1 , n_redundant= 0, n_repeated= 0)
plt.scatter(x,y, c = y, cmap = "rainbow")
plt.title("scatterplot of logistic regression")
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1)
lr = LogisticRegression()
lr.fit(x_train,y_train)
print (lr.coef_)
print(lr.intercept_)
y_prediction = lr.predict(x_test)
print(y_prediction)