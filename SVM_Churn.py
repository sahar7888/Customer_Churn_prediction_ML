#  Python code to model Customer Churn with Decision Tree
#  Data Source: https://www.kaggle.com/blastchar/telco-customer-churn

# %%  Import basic libraries, read data and
###   do preliminary cleaning (irrelevant cols, missing values)

import pandas as pd  # For data handling
from os import chdir  # To change directory
from sklearn.model_selection import train_test_split  # To split train/test
from sklearn.svm import SVC  # For SVM
from sklearn.metrics import accuracy_score  # Accuracy
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn import metrics  # & other metrics

##  Change path and Read data from csv file

chdir("./data")
churn = pd.read_csv('churn_data.csv')

##  Remove the customerID columns (not related to churn)
del churn['customerID']

##  'TotalCharges' has 11 blank values.  We change them to '0'
##          and  convert TotalCharges --> 'float'
churn['TotalCharges'] = churn['TotalCharges'].str.replace(' ', '0')
churn['TotalCharges'] = churn['TotalCharges'].astype(float)

# %%  Next phase of Preprocessing ::
###   Map all categoric columns to numeric values (0, 1, 2, etc.)

churnum = churn.copy()

map_num = {"gender": {"Male": 1, "Female": 0},
           "Partner": {"Yes": 1, "No": 0},
           "Dependents": {"Yes": 1, "No": 0},
           "PhoneService": {"Yes": 1, "No": 0},
           "MultipleLines": {"Yes": 2, "No": 1, "No phone service": 0},
           "InternetService": {"Fiber optic": 2, "DSL": 1, "No": 0},
           "OnlineSecurity": {"Yes": 2, "No": 1, "No internet service": 0},
           "OnlineBackup": {"Yes": 2, "No": 1, "No internet service": 0},
           "DeviceProtection": {"Yes": 2, "No": 1, "No internet service": 0},
           "TechSupport": {"Yes": 2, "No": 1, "No internet service": 0},
           "StreamingTV": {"Yes": 2, "No": 1, "No internet service": 0},
           "StreamingMovies": {"Yes": 2, "No": 1, "No internet service": 0},
           "Contract": {"Month-to-month": 1,
                        "One year": 12,
                        "Two year": 24},
           "PaperlessBilling": {"Yes": 1, "No": 0},
           "PaymentMethod": {"Electronic check": 1,
                             "Mailed check": 0,
                             "Bank transfer (automatic)": 2,
                             "Credit card (automatic)": 3},
           "Churn": {"Yes": 1, "No": 0}
           }

churnum.replace(map_num, inplace=True)

###  Convert  data frame  to  arrays... and
###    split into  Train/Test datasets

X = churnum.values[:, :19]
y = churnum.values[:, 19]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=39)

# %%  Building the SVM model and Predicting

model_svm = SVC(kernel='poly',  ##  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
                degree=1,
                max_iter=1000,
                random_state=39)

model_svm.fit(X_train, y_train)

##  Predict  on  training  &  test  data  sets

y_pren_svm = model_svm.predict(X_train)
y_pret_svm = model_svm.predict(X_test)

##  Training set results

cm_svm = confusion_matrix(y_train, y_pren_svm)
accu_svm = accuracy_score(y_train, y_pren_svm)
sens_svm = cm_svm[0, 0] / float(cm_svm[0, 0] + cm_svm[0, 1])
spec_svm = cm_svm[1, 1] / float(cm_svm[1, 0] + cm_svm[1, 1])
fpr, tpr, thsld = metrics.roc_curve(y_train, y_pren_svm, pos_label=1)


print("Confusion Matrix for SVM_train : \n", cm_svm)
print("Accuracy for SVM_train is : ", round(accu_svm * 100, 2), "%")
print("Sensitivity for SVM_train = ", round(sens_svm * 100, 2), "%")
print("Specificity for SVM_train = ", round(spec_svm * 100, 2), "%")
print("The AUC-ROC for SVM_train : ", round(metrics.auc(fpr, tpr) * 100, 2), "%")

##  Test set results

cm_svm = confusion_matrix(y_test, y_pret_svm)
accu_svm = accuracy_score(y_test, y_pret_svm)
sens_svm = cm_svm[0, 0] / float(cm_svm[0, 0] + cm_svm[0, 1])
spec_svm = cm_svm[1, 1] / float(cm_svm[1, 0] + cm_svm[1, 1])
fpr, tpr, thsld = metrics.roc_curve(y_test, y_pret_svm, pos_label=1)

print("Confusion Matrix for SVM_test  : \n", cm_svm)
print("Accuracy for SVM_test is  : ", round(accu_svm * 100, 2), "%")
print("Sensitivity for SVM_test  = ", round(sens_svm * 100, 2), "%")
print("Specificity for SVM_test  = ", round(spec_svm * 100, 2), "%")
print("The AUC-ROC for SVM_test  : ", round(metrics.auc(fpr, tpr) * 100, 2), "%")
