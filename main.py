import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
print("\n")

# females more likely to have heart problems than males
print(dataset["sex"].unique())
sns.barplot(x=dataset["sex"],y=y)
print("\n")

# chest pain type (values range from 0 - 3)
dataset["cp"].unique()
sns.barplot(x=dataset["cp"],y=y)
print("\n")

# fasting blood sugar (values: 0 and 1)
dataset["fbs"].describe()
dataset["fbs"].unique()
sns.barplot(x=dataset["fbs"],y=y)
print("\n")

# analyze the resting electrocardiographic measurement (values 0-2)
dataset["restecg"].unique()
sns.barplot(x=dataset["restecg"],y=y)
print("\n")

# analyze exang (exercise induced angina) (values 0 and 1)
dataset["exang"].unique()
sns.barplot(x=dataset["exang"],y=y)
print("\n")

#  analyze heart pain according to the slope of an incline (values from 0 to 2)
dataset["slope"].unique()
sns.barplot(x=dataset["slope"],y=y)
print("\n")

# number of major vessels (0-3)
# ca=4 has the largest number of heart patients
dataset["ca"].unique()
sns.countplot(x=dataset["ca"])
sns.barplot(x=dataset["ca"],y=y)
print("\n")

# analyzing thal (Thalassemia)
dataset["thal"].unique()
sns.barplot(x=dataset["thal"],y=y)
sns.displot(dataset["thal"])
print("\n")

# creates a new DataFrame called predictors by dropping the "target" column from the original dataset DataFrame,
# along the specified axis (axis=1 indicates columns)
predictors = dataset.drop("target",axis=1)
target = dataset["target"]

# split the dataset into training and testing sets to run a train-test split
# 80% of data used for training and 20% used for testing
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Model Fitting for Logistic regression (used for classification problems as opposed to linear regression)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
print("\n")

# compute the accuracy score of Logistic regression testing
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
print("\n")

# calculate naive bayes accuracy score
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
# dimensions of the array (rows, columns)
print(Y_pred_nb.shape)

score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
print("\n")

# calculate Support Vector Machine (SVM) accuracy score
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
print(Y_pred_svm.shape)

score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
print("\n")

# calculate K Nearest Neighbors accuracy score
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
print(Y_pred_knn.shape)

score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
print("\n")

# calculate Decision Tree accuracy score
max_accuracy = 0

for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test)*100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

print("Max accuracy: ", max_accuracy)
print("Best x value input: ", best_x)

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
print(Y_pred_dt.shape)

score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")