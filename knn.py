import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sl
dataset=pd.read_csv("diabetes.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
from  sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))
