import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## Load data
cancer = load_breast_cancer()
print(cancer.keys())
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())

## Scale the data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

## Setup PCA model
pca = PCA(n_components=2)
pca.fit(scaled_data)

## Transform to first 2 principle components
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

## Plot malignant and benign components
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()

## Interpret the components
print(pca.components_)

## Heatmap the features to better understand the data
df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')
plt.show()