import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

## Sets a random seed
np.random.seed(0)

## Load the data and create the dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

## Add the species target column to the dataframe
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

## Create test and train data (This just generates a random number between 0 and 1, 75% of the data is train data)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
print(df.head())

## Split the data
train, test = df[df['is_train']==True], df[df['is_train']==False]

print("Number of observations in the training data:", len(train))
print("Number of observations in the test data:", len(test))

## Create a list of feature columns names
features = df.columns[:4]
print(features)

## Convert each species name into digits
## Creates a unique value for each specific species
y = pd.factorize(train['species'])[0]
print(y)

## Create the random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

## Train the classifier (Features is data, y is the label)
clf.fit(train[features], y)

## Apply the trained classifier to the test and predict the first 20
predictions = clf.predict_proba(test[features])[0:20]
print(predictions)

## Map names for each prediction
preds = iris.target_names[clf.predict(test[features])]
print(preds[0:5])

print(test['species'].head())

## Create confusion matrix
confmatrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
print(confmatrix)