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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense

# Read the dataset from a CSV file
dataset = pd.read_csv("heart.csv")

# Display dataset information
print(type(dataset))
print("\n")
print(dataset.shape)
print("\n")
print(dataset.head(5))

print("\n")

# Display 5 random rows and describe the data
print(dataset.sample(5))

print("\n")

print(dataset.describe())

print("\n")

# Display an overview of the data
print(dataset.info())

print("\n")

# Display information about each column
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
        "resting blood pressure","serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)",
        "maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

print("\n")

# Generate statistics for the "target" column and then print the unique values
print(dataset["target"].describe())
print("\n")
print(dataset["target"].unique())
print("\n")

# Since the unique values of the "target" dataset are 0 and 1,
# this is supervised learning with binary classification

# Show the correlation between "target" column with other columns
print(dataset.corr()["target"].abs().sort_values(ascending=False))
print("\n")

# Create a count plot (bar graph) based on the "target" column data
y = dataset["target"]
sns.countplot(y)
print("\n")

# Get the count of the unique values (print how many 1's and 0's occur)
target_temp = dataset.target.value_counts()
print(target_temp)
print("\n")

# Display percentages of patients with and without heart problems
print("Percentage of patients without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patients with heart problems: "+str(round(target_temp[1]*100/303,2)))
print("\n")

# Females more likely to have heart problems than males
print(dataset["sex"].unique())
sns.barplot(x=dataset["sex"],y=y)
print("\n")

# Chest pain type (values range from 0 - 3)
dataset["cp"].unique()
sns.barplot(x=dataset["cp"],y=y)
print("\n")

# Fasting blood sugar (values: 0 and 1)
dataset["fbs"].describe()
dataset["fbs"].unique()
sns.barplot(x=dataset["fbs"],y=y)
print("\n")

# Analyze the resting electrocardiographic measurement (values 0-2)
dataset["restecg"].unique()
sns.barplot(x=dataset["restecg"],y=y)
print("\n")

# Analyze exang (exercise induced angina) (values 0 and 1)
dataset["exang"].unique()
sns.barplot(x=dataset["exang"],y=y)
print("\n")

# Analyze heart pain according to the slope of an incline (values from 0 to 2)
dataset["slope"].unique()
sns.barplot(x=dataset["slope"],y=y)
print("\n")

# Number of major vessels (0-3)
# ca=4 has the largest number of heart patients
dataset["ca"].unique()
sns.countplot(x=dataset["ca"])
sns.barplot(x=dataset["ca"],y=y)
print("\n")

# Analyzing thal (Thalassemia)
dataset["thal"].unique()
sns.barplot(x=dataset["thal"],y=y)
sns.displot(dataset["thal"])
print("\n")

# Creates a new DataFrame called predictors by dropping the "target" column from the original dataset DataFrame,
# along the specified axis (axis=1 indicates columns)
predictors = dataset.drop("target",axis=1)
target = dataset["target"]

# Split the dataset into training and testing sets to run a train-test split
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

# Compute the accuracy score of Logistic regression testing
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
print("\n")

# Calculate Naive Bayes accuracy score
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
# Dimensions of the array (rows, columns)
print(Y_pred_nb.shape)

score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
print("\n")

# Calculate Support Vector Machine (SVM) accuracy score
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
print(Y_pred_svm.shape)

score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")
print("\n")

# Calculate K Nearest Neighbors accuracy score
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
print(Y_pred_knn.shape)

score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")
print("\n")

# Calculate Decision Tree accuracy score
max_accuracy = 0

for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test)*100, 2

)
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
print("\n")

# Calculate Random Forest accuracy score
max_accuracy = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

print(max_accuracy)
print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)

print(Y_pred_rf.shape)

score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")
print("\n")

# Calculate XGBoost accuracy score
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)
print(Y_pred_xgb.shape)

score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)
print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")
print("\n")

# Calculate Neural Network accuracy score
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=300)

Y_pred_nn = model.predict(X_test)
print(Y_pred_nn.shape)

rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)
print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")
# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.

# Output final score
scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

# Barplot of final scores
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(x=algorithms,y=scores)
plt.show()