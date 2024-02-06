import numpy as np 
import  pandas as pd
import  matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#for reading data from csv file
df=pd.read_csv("Ninapro_DB1.csv")

#dropping certain columns with no use
columns_to_drop=['stimulus','restimulus','repetition','rerepetition','subject']
df=df.drop(columns=columns_to_drop)

#setting target and variable
target_column='exercise'
X=df.drop(columns=[target_column])
y=df[target_column]

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Creating a classifier object
rf_classifier=RandomForestClassifier(n_estimators=100,random_state=42)

#Training the model using the training data
rf_classifier.fit(X_train,y_train)

#Making predictions on the test set
y_pred=rf_classifier.predict(X_test)

#Evaluation
print('Accuracy:',accuracy_score(y_test,y_pred))
print('\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
print('\n')
print('Classification Report:\n',classification_report(y_test,y_pred))







    









































