import time
start=time.time()

import os
path=os.path.abspath(os.curdir)

import pandas as pd

import csv
pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

import numpy as np

# Converting xlsx to csv
import glob
Data=glob.glob(path+'/Train_dataset.xlsx')
for excel in Data:
	sheet=excel.split('.')[0]+'.csv'
	df=pd.read_excel(excel)
	df.to_csv(sheet)

Train=pd.read_csv(path+'/Train_dataset.csv',low_memory=False)
#Turn off the warnings
import warnings
warnings.filterwarnings("ignore")

# Drop any column with more than 50% missing values
half_count=len(Train)/2
Train= Train.dropna(thresh=half_count,axis=1) 

#Column which need to be converted to float for fitting in the model
predictor_columns=['Region','Gender','Married','Occupation','Mode_transport','comorbidity','Pulmonary score','cardiological pressure']

#Splitting datset into Target and data
y=Train['Infect_Prob']

x1=Train.drop('Infect_Prob',axis=1)
df=x1.drop('Designation',axis=1)
x=df.drop('Name',axis=1)

#Converting string to float
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in predictor_columns:
    le.fit(x[column].astype(str))
    x[column] = le.transform(x[column].astype(str))
    
#Filling NULL, Nan and infinity values
from sklearn.metrics import mean_absolute_error
x=x.fillna(x.mean())
y=y.fillna(y.mean())

#Splitting the data into training and testing  dataset
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=0.20,random_state=20)

#Transorming data for KNN model
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train_X)
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)

#Training the model 
from sklearn.svm import SVR
model=SVR(kernel='rbf')
model.fit(train_X,train_Y)
print("Model fit succesful")
pred=model.predict(test_X)

#################################################################

#Uploading test dataset
Data=glob.glob(path+'/Test_dataset.xlsx')
for excel in Data:
	sheet=excel.split('.')[0]+'.csv'
	df=pd.read_excel(excel)
	df.to_csv(sheet)

prediction=pd.read_csv(path+'/Test_dataset.csv',low_memory=False)
Test=prediction.drop('Designation',axis=1)
X=Test.drop('Name',axis=1)

#Converting from string to float
for col in predictor_columns:
    le.fit(X[col].astype(str))
    X[col] = le.transform(X[col].astype(str))
X=X.fillna(X.mean())

#Transorming data for  model
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
Test=scaler.transform(X)

#Prediction using given test dataset
Predicted_Infect_prob=model.predict(Test)


Infect_Prob=pd.DataFrame(Predicted_Infect_prob)
res=pd.DataFrame(Infect_Prob)
res.index=X['people_ID']
res.columns=['Infect_Prob']
res.to_csv(path+"/output_file_01.csv")
end=time.time()
print('Exec. Time',end-start)