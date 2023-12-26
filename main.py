import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

dataset = pd.read_csv("heart.csv")

# print out the dataset type (should be dataframe), the dataset shape (number of rows and columns), and the first 5 rows
print(type(dataset))
print(dataset.shape)
print(dataset.head(5))

# print 5 random rows and describe the data
print(dataset.sample(5))
print(dataset.describe())