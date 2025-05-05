# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import svm
# from sklearn.metrics import accuracy_score

# # loading the data from csv file to a Pandas DataFrame
# import os
# base_path = os.path.dirname(__file__)
# data_path = os.path.join(base_path, '../data/parkinsons.data')
# print("Loading data from:", data_path)
# parkinsons_data = pd.read_csv(data_path)

# X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
# Y = parkinsons_data['status']

# scaler = StandardScaler()


# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=2)

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.fit_transform(X_test)

# model = svm.SVC(kernel='linear')

# # training the SVM model with training data
# model.fit(X_train, Y_train)

# # accuracy score on training data
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# print('Accuracy score of training data : ', training_data_accuracy)

# # accuracy score on training data
# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# print('Accuracy score of test data : ', test_data_accuracy)


# def run_app(df):

#     store_df = df
#     uploaded_patient_list = []
#     prediction_list = []
#     patient_id_list = []
#     patient_id_list.append(df['name'].tolist())
#     patient_id_list = patient_id_list[0]

#     X = df.drop(columns=['name'], axis=1)

#     for index, rows in X.iterrows():

#         uploaded_patient_list. append(X.loc[index, :].values.tolist())

#     new_X = scaler.fit_transform(uploaded_patient_list)

#     X_prediction = model.predict(new_X)

#     store_df['status'] = X_prediction

#     for i in X_prediction:

#         if i == 1:
#             prediction_list.append("The Patient has Parkinson Disease")

#         if i == 0:

#             prediction_list.append(
#                 "The Patient does not have Parkinson Disease")

#     return_csv = pd.DataFrame(
#         {"Patient ID": patient_id_list, "Prediction": prediction_list})

#     return return_csv, store_df


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.tree import plot_tree

# Load dataset
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, '../data/parkinsons.data')
df = pd.read_csv(data_path)

# Exploratory Data Analysis (Optional but useful during dev)
print(df.head())
print(df.info())
print(df.describe())
print("Null values:", df.isnull().sum())
print("Target Distribution:\n", df['status'].value_counts())


# Features and labels
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# SVM
# Train linear SVM
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test_scaled)
y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Parkinsons'],
            yticklabels=['Healthy', 'Parkinsons'])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.plot(fpr_svm, tpr_svm, label=f'SVM ROC (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("SVM ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# RANDOM FOREST

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n",
      classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Healthy', 'Parkinsons'],
            yticklabels=['Healthy', 'Parkinsons'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf,
         label=f'RF ROC (AUC = {roc_auc_rf:.2f})', color='darkgreen')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Random Forest ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Visualize one decision tree
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns,
          class_names=['Healthy', 'Parkinsons'], filled=True,
          max_depth=3, rounded=True)
plt.title("Random Forest - Single Tree (Depth = 3)")
plt.show()


# Example patient data
input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168,
              0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339,
              26.77500, 0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

input_np = np.asarray(input_data).reshape(1, -1)

try:
    std_input = scaler.transform(input_np)
    prediction = svm_model.predict(std_input)  # or rf_model.predict(std_input)

    print("Prediction Result:",
          "HAS Parkinson’s Disease" if prediction[0] == 1 else "Does NOT have Parkinson’s Disease")
except Exception as e:
    print("Error during prediction:", e)


def run_app(df):

    store_df = df
    uploaded_patient_list = []
    prediction_list = []
    patient_id_list = []
    patient_id_list.append(df['name'].tolist())
    patient_id_list = patient_id_list[0]

    X = df.drop(columns=['name', 'status'], axis=1)

    for index, rows in X.iterrows():

        uploaded_patient_list. append(X.loc[index, :].values.tolist())

    new_X = scaler.fit_transform(uploaded_patient_list)

    X_prediction = rf_model.predict(new_X)

    store_df['status'] = X_prediction

    for i in X_prediction:

        if i == 1:
            prediction_list.append("The Patient has Parkinson Disease")

        if i == 0:

            prediction_list.append(
                "The Patient does not have Parkinson Disease")

    return_csv = pd.DataFrame(
        {"Patient ID": patient_id_list, "Prediction": prediction_list})

    return return_csv, store_df
