import numpy as np
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1,1)
y = np.array([0,1,0,0,1,1,1,1,1,1])
model = LogisticRegression(solver = "liblinear",C = 10.0, random_state = 0)
model.fit(x,y)
prediction_probalility = model.predict_proba(x)
y_prediction = model.predict(x)
score1 = model.score(x,y)
confision_matrics1 = confusion_matrix(y,y_prediction)
report = classification_report(y,y_prediction)
print(f"x:{x} y: {y}")
print(f"the prediction probability is {prediction_probalility}")
print(f"the y_prediction is {y_prediction}")
print(f"the score1 is {score1}")
print(f"the confusion matrics1 is {confision_matrics1}")