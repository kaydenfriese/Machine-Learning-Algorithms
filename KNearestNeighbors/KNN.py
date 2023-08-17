import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

## Load the dataset
dataset = pd.read_csv(r'C:\Users\kayde\OneDrive\Documents\GitHub\Machine-Learning-Algorithms\KNearestNeighbors\diabetes.csv')
print(len(dataset))
print(dataset.head())

## Replace Zeros in data
zeros_not_expected = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zeros_not_expected:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

## Split the data
X = dataset.iloc[:, 0:8] ## All rows (:) 0 to 8 (0:8)
y = dataset.iloc[:, 8] ## Just the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

## Scale the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Define the model
## 11 neighbors beacuse sqrt(len(y)) = 12 (12-1 = 11) 
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
classifier.fit(X_train, y_train)

## Predict the test set
y_pred = classifier.predict(X_test)

## Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))