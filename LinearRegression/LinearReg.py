## Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

## Import data and get X and Y variables (Independant and Dependant Respectively)
companies = pd.read_csv(r'C:\Users\kayde\source\repos\MachineLearningAlgos\LinearRegression\1000_Companies.csv')

X = companies.iloc[:, :-1].values
Y = companies.iloc[:, 4].values

## Print all data
print(companies.head())
## Print independant data
print(X)
## Print dependant data
print(Y)

## Builds correlation matrix : 1 = more correlation 0 = less
plt.figure(figsize=(10, 7))
## Only include columns with numbers bc of error
numeric_companies = companies.select_dtypes(include=[np.number])
sns.heatmap(numeric_companies.corr(), annot=True, cmap="coolwarm")
plt.show()

## Encode the categorical data
column_transformer = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(drop='first'), [3])  # We're using drop='first' to avoid the dummy variable trap
    ],
    remainder='passthrough'
)

X = column_transformer.fit_transform(X)
X = X[:, 1:] ## Removes dummy data

## Create the train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

## Initialize the model
regressor = LinearRegression()

## Fit with the training data
regressor.fit(X_train, y_train)

## Predict Results
y_pred = regressor.predict(X_test)
print(y_pred)

## Prints the coefficient
print(regressor.coef_)

## Prints the Y intercept
print(regressor.intercept_)

## Evaluate the model
print(r2_score(y_test, y_pred))

## Closer to 1 = better model, this model has a r2 score of .91 so it is accurate
