import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


#You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p på£3.99
#Miles and smiles r made frm same letters but do u know d difference..? smile on ur face keeps me happy even though I am miles away from u.. :-)keep smiling.. Good nyt


ssc = pd.read_csv('spam.csv',encoding="ISO-8859-1")
#print(ssc)
ssc = ssc.where(pd.notnull(ssc),'')
#print(ssc)


ssc.loc[ssc['v1']=='spam','v1']=0
ssc.loc[ssc['v1']=='ham','v1']=1
#print(ssc)


#print(ssc.columns)

ssc['Message'] = ssc['v2']+ssc['Unnamed: 2']+ssc['Unnamed: 3']+ssc['Unnamed: 4']
#print(ssc.columns)

ssc=ssc.drop(['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
ssc.rename(columns={'v1':'Category'},inplace=True)
#print(ssc.columns)

a = ssc['Message']
b = ssc['Category']

train_a,test_a,train_b,test_b = train_test_split(a,b,test_size=0.2)
feature_extraction = TfidfVectorizer(dtype=np.float32)
train_a = feature_extraction.fit_transform(train_a)
test_a = feature_extraction.transform(test_a)
train_b = train_b.astype('int')
test_b = test_b.astype('int')

#print(train_a)

model = LogisticRegression()
model.fit(train_a,train_b)
predict = model.predict(test_a)
#print(predict)
print('Accuracy of model = ',accuracy_score(test_b,predict))
ans = 1
while(ans):
    x=np.array([input("Enter message for predicting : ")])
    x=feature_extraction.transform(x)
    pred = model.predict(x)
    if pred==1:
        print("ham")
    else:
        print("spam")
    ans = float(input("Predict again (yes = 1,no=0) : "))