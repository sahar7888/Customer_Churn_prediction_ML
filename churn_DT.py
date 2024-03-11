

"""Ref: Machine Learning With Python: Predicting Customer Churn (Udemy)
"""

#  Python code to model Customer Churn with Decision Tree
#  Data Source: https://www.kaggle.com/blastchar/telco-customer-churn


""" Using Decision Tree"""
import pandas as pd
from os import chdir
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

# %%  Visualizing the Decision Tree

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus



chdir("./data")
churn = pd.read_csv('churn_data.csv')

##  Remove the customerID columns (not related to churn)
del churn['customerID']

##  'TotalCharges' has 11 blank values.  We change them to '0'
##          and  convert TotalCharges --> 'float'
churn['TotalCharges'] = churn['TotalCharges'].str.replace(' ','0')
churn['TotalCharges'] = churn['TotalCharges'].astype(float)

#%%  Next phase of Preprocessing ::
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
y = churnum.values[:,  19]


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 39)

# %%  Building the Tree and Predicting

# tree_depth = 3

for tree_depth in range(2, 15):
    model_dt = DecisionTreeClassifier(criterion="gini",
                                      random_state=39,
                                      max_depth=tree_depth,
                                      min_samples_leaf=50)
    model_dt.fit(X_train, y_train)

    ##  Predict  on  training  &  test  data sets
    y_pren_dt = model_dt.predict(X_train)
    y_pret_dt = model_dt.predict(X_test)

    ##  Training set results
    #    cm_dt = confusion_matrix(y_train, y_pren_dt)
    #
    #    sens_dt = float(cm_dt[0,0])/(cm_dt[0,0]+cm_dt[0,1])
    #    spec_dt = float(cm_dt[1,1])/(cm_dt[1,0]+cm_dt[1,1])
    #
    #    fpr, tpr, thsld = metrics.roc_curve(y_train, y_pren_dt, pos_label=1)
    #
    #    print tree_depth, round(accuracy_score(y_train, y_pren_dt)*100, 2), \
    #          round(sens_dt*100, 2), round(spec_dt*100, 2), \
    #          round(metrics.auc(fpr, tpr)*100, 2)               ##  comment one print

    ##  Test set results

    cm_dt = confusion_matrix(y_test, y_pret_dt)

    sens_dt = float(cm_dt[0, 0]) / (cm_dt[0, 0] + cm_dt[0, 1])
    spec_dt = float(cm_dt[1, 1]) / (cm_dt[1, 0] + cm_dt[1, 1])

    fpr, tpr, thsld = metrics.roc_curve(y_test, y_pret_dt, pos_label=1)

    print(tree_depth, round(accuracy_score(y_test, y_pret_dt) * 100, 2), \
        round(sens_dt * 100, 2), round(spec_dt * 100, 2), \
        round(metrics.auc(fpr, tpr) * 100, 2) ) ##  comment one print

# %%  Visualizing the Decision Tree


dot_data = StringIO()
export_graphviz(model_dt, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=list(churnum.iloc[:, :19]),
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

nodes = graph.get_node_list()

for node in nodes:
    if node.get_label():
        values = [int(float(ii)) for ii in node.get_label().split('value = [')[1].split(']')[0].split(',')]
        values = [int(255 * v / sum(values)) for v in values]
        #  make colors 50% lighter
        tt = 0.5
        rr = 255 - int((255 - values[1]) * tt)
        gg = 255 - int((255 - 128) * tt)
        bb = 255 - int((255 - values[0]) * tt)
        color = '#{:02x}{:02x}{:02x}'.format(rr, gg, bb)
        node.set_fillcolor(color)

graph.write_pdf('churn_DT-3.pdf')
Image(graph.create_png())


