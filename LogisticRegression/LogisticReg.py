from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

## Load the digit images
digits = load_digits()

## Make sure the Data and Labels have the same shape
print("Image Data Shape", digits.data.shape)
print("Image Label Shape", digits.target.shape)

## This compiles 1 of each number up to 5 into a pyplot chart to see what they look like
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title("Training %i\n" % label, fontsize=20)
plt.show()

## digits.data is the data shape, digits.target contains the labels 
## Shape is X, Labels is Y
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

## Make sure the shapes are the same
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

## Initialize the model and fit with training data
logisticReg = LogisticRegression()
logisticReg.fit(x_train, y_train)

## Return a numpy array, test for one image
## Reshape when only one sample using .reshape(1,-1)
print(logisticReg.predict(x_test[0].reshape(1,-1)))

## Predict 10 samples
print(logisticReg.predict(x_test[0:10]))

## Create predictions
predictions = logisticReg.predict(x_test)

## Print the score of the model
score = logisticReg.score(x_test, y_test)
print(score)

## Create the confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

## Create the heatmap and show with pyplot
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

## Get the wrong index's and show the predicted vs actual on a graph
index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict==actual:
        classifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,3))
for plotindex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1, 4, plotindex+1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]),  fontsize=20)
plt.show()