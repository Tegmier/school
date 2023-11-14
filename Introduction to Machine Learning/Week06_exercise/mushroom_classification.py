from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd

# Read the csv file from local
data = np.array(pd.read_csv('d:\code\school\Introduction to Machine Learning\Week06_exercise\mushrooms.csv'))

print('Shape of Dataset: ', data.shape)
print('datatype of Dataset: ', data.dtype)

# Data preprocessing
for row in range(len(data)):
    for column in range(len(data[row])):
        element = data[row, column]
        data[row, column] = ord(element) - ord('a') + 1

label = data[:,0].astype(int)

for i in range(len(label)):
    if label[i] > 5:
        label[i] = 1
    else:
        label[i] = 0
features = data[:,1:].astype(int)

# Training dataset and Test dataset preparation
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=42)

# Logistic Regression
# mushroomsRegression = LogisticRegression(max_iter = 5000)
mushroomsRegression = LogisticRegression(solver = 'liblinear', max_iter = 5000)
mushroomsRegression.fit(X_train, y_train)

mushroomsPredict = mushroomsRegression.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, mushroomsPredict)
conf_marix = confusion_matrix(y_test, mushroomsPredict)
coefficients = mushroomsRegression.coef_

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_marix)
print("Coefficients:", coefficients)


