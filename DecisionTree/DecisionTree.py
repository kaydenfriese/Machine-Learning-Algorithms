import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

## Load Data and remove the sum column because it is not needed
balanced_data = pd.read_csv(r'C:\Users\kayde\OneDrive\Documents\GitHub\Machine-Learning-Algorithms\DecisionTree\Loan Repayment.csv', sep=",", header=0)
balanced_data = balanced_data.drop(columns='sum')

## Get length and shape of the dataset
print("Dataset Length::", len(balanced_data))
print("Dataset Shape::", balanced_data.shape)
print(balanced_data.head())

## Separate the target data
## X is numerical data
## Y is label data
X = balanced_data.values[:, 1:4]
Y = balanced_data.values[:, 4]

## Split dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

## Perform training with entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)

## Test the model and predict
y_pred_en = clf_entropy.predict(x_test)
print(y_pred_en)

## Print Accuracy
print("Accuracy: {}".format(accuracy_score(y_test, y_pred_en)*100))
