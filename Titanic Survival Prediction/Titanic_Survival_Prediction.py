import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


tsp = pd.read_csv("titanic.csv")
#print(tsp)
labelencoder = LabelEncoder()
tsp['Sex'] = labelencoder.fit_transform(tsp['Sex'])
#print(tsp['Sex'])


#print(tsp.isnull().sum())
del tsp['Cabin']
tsp=tsp.dropna()
#print(tsp.isnull().sum())

#print(tsp.columns)


x=tsp[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y=tsp.iloc[:,1]
#print(x)
#print(y)


train_a,test_a,train_b,test_b = train_test_split(x,y,test_size=0.2)

model_lr = LogisticRegression(random_state=0)
model_lr.fit(train_a.values,train_b)
pred_lr = model_lr.predict(test_a.values)
#print(pred_lr,test_b)
print('Accuracy of model = ',metrics.accuracy_score(pred_lr,test_b)*100,'%')

#predicting manually
flag=1
while(flag==1):
    Pclass=int(input("Enter Pclass : "))
    Sex=int(input("Enter Sex : "))
    Age=float(input("Enter Age : "))
    SibSp=int(input("Enter SibSp : "))
    Parch=int(input("Enter Parch : "))
    Fare=float(input("Enter Fare : "))
    test_m = np.array([[Pclass,Sex,Age,SibSp,Parch,Fare]])
    pred = model_lr.predict(test_m)
    if(pred == 0):
        print("Sorry they didn't survived :(")
    else:
        print("Survived")
    flag=int(input("Wanna predict again ?"))