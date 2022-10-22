import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv(r'C:\Users\User\Desktop\prep\cv_projects\Project\p3\data\parkinsons.data')

X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

scaler = StandardScaler()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

model = svm.SVC(kernel='linear')

# training the SVM model with training data
model.fit(X_train, Y_train)

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data : ', test_data_accuracy)



def run_app(df):

    store_df =df
    uploaded_patient_list=[]
    prediction_list=[]
    patient_id_list=[]
    patient_id_list.append(df ['name'].tolist())
    patient_id_list = patient_id_list[0]


    X = df.drop(columns=['name'], axis=1)

    for index, rows in X.iterrows():

        uploaded_patient_list. append(X.loc[index, :].values.tolist())



    new_X = scaler.fit_transform(uploaded_patient_list)

    X_prediction = model.predict(new_X)

    store_df['status'] = X_prediction

    for i in X_prediction:

        if i==1 :
            prediction_list.append("The Patient has Parkinson Disease")

        if i==0 :

            prediction_list.append("The Patient does not have Parkinson Disease")


    return_csv = pd.DataFrame({"Patient ID": patient_id_list, "Prediction" : prediction_list})


    return return_csv, store_df

    





