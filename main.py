import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

dataset = pd.read_csv("heart.csv")

# print out the dataset type (should be dataframe), the dataset shape (number of rows and columns), and the first 5 rows
print(type(dataset))
print("\n")
print(dataset.shape)
print("\n")
print(dataset.head(5))

print("\n")

# print 5 random rows and describe the data
print(dataset.sample(5))

print("\n")

print(dataset.describe())

print("\n")

# give an overview of the data
print(dataset.info())

print("\n")

# print out example data
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

print("\n")

# generate statistics for the "target" column and then print the unique values
print(dataset["target"].describe())
print("\n")
print(dataset["target"].unique())
print("\n")

# since the unique values of the "target" dataset are 0 and 1,
# this is supervised learning with binary classification


# show the correlation between "target" column with other columns
print(dataset.corr()["target"].abs().sort_values(ascending=False))
print("\n")

# create a count plot (bar graph) based on the "target" column data
y = dataset["target"]
sns.countplot(y)
print("\n")

# get the count of the unique values (print how many 1's and 0's occur)
target_temp = dataset.target.value_counts()
print(target_temp)
print("\n")

print("Percentage of patients without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patients with heart problems: "+str(round(target_temp[1]*100/303,2)))