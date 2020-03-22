import time
start=time.time()

import os
path=os.path.abspath(os.curdir)

import pandas as pd

import csv
pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

import numpy as np

# Converting xlsx to csv for Training data
import glob
Data=glob.glob(path+'/Train_dataset.xlsx')
for excel in Data:
	sheet=excel.split('.')[0]+'.csv'
	df=pd.read_excel('Train_dataset.xlsx',sheet_name='Train_dataset')
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


#################################################################

#Uploading Test dataset
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

#Transorming data for Prediction
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
Test=scaler.transform(X)



#####################################################################
#converting excel to csv for Diuresis prediction
for excel in Data:
	sheet1='Diuresis_TS.csv'
	df=pd.read_excel(path+'/Train_dataset.xlsx',sheet_name='Diuresis_TS')
	df.to_csv(sheet1)

# load dataset
series = pd.read_csv('Diuresis_TS.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# split data into train and test
Y = series.values
train, test = Y[0:-10713], Y[-10713:]

# walk-forward validation
history = [x for x in train]
predictions =[]
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])

############################

# #Replacing the deuresis data in training dataset
D_X=x.drop('Diuresis',axis=1)

#training model for updated dataset for INFECT_PROB on 27th March
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)

#Transorming data for model
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# Training the model 
from sklearn.svm import SVR
model=SVR(kernel='rbf')
model.fit(x_train,y_train)
print("Model fit succesful")
Diuresis_Infect_prob=model.predict(Test)


Infect_Prob=pd.DataFrame(Diuresis_Infect_prob)
D_res=pd.DataFrame(Infect_Prob)
D_res.index=X['people_ID']
D_res.columns=['Infect_Prob']
D_res.to_csv(path+"/output_file_02.csv")

end=time.time()
print('Exec. Time',end-start)




