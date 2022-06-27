import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import time
df = pd.read_csv('creditcard.csv')


train , test = train_test_split(df, test_size = 0.3, stratify=df["Class"])

y_train = train["Class"].copy()
X_train = train.drop("Class", axis=1).copy()
y_test = test["Class"].copy()
X_test = test.drop("Class", axis=1).copy()

print("Train: {}  Test: {}".format(X_train.shape,X_test.shape))




print (X_train)

def train_model_and_pred(model, X_train, y_train):

    from sklearn.metrics import confusion_matrix  , classification_report
    time_start = time.perf_counter() 
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    #precision = precision_score(y_train, train_predictions)
    #recall = recall_score(y_train, train_predictions)
    f1 = f1_score(y_test, predictions,average='macro')
    print("macro_F1 score ", f1)
    time_end = time.perf_counter()
    
    total_time = time_end-time_start
    print("Время: %4.3f" % (total_time))
    
    c = confusion_matrix(y_test, predictions)
    print(classification_report(y_test, predictions))
    print(c)



def bag_():
    from sklearn.metrics import confusion_matrix  , classification_report
    from sklearn.ensemble import BaggingClassifier
    ok=BaggingClassifier(n_estimators=20, random_state=421)
    params = {'n_estimators': [110,29], 'bootstrap': [True]}
    clf = GridSearchCV(ok, params, scoring='f1', verbose=50, n_jobs=4)
    clf.fit(X_train, y_train)
    best_clf = clf.best_estimator_
    
    Y_pred = best_clf.predict(X_test)
    print(classification_report(y_test, Y_pred))
    f1 = f1_score(y_test, Y_pred,average='macro')
    print("macro_F1 score ", f1)
    print(confusion_matrix(y_test, Y_pred))


def tree():
    from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    params = {'n_estimators': [100],
            'max_features': [6],
            'min_samples_split':[20],
            }
    F = RandomForestClassifier(bootstrap=True, random_state=42, n_jobs=4,max_depth=16)
    model = GridSearchCV(F,params, n_jobs=4, verbose=1)
    model.fit(X_train, y_train)

    # метрики результата предсказания
    Y_pred = model.predict(X_test)
    f1 = f1_score(y_test, Y_pred,average='macro')
    print("macro_F1 score ", f1)
    print(classification_report(y_test, Y_pred))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, Y_pred))



def run_GBM():  
    

    model=lgbm.LGBMClassifier(boosting_type='dart',device='gpu',
                           
                             feature_fraction = 0.5,
                              feature_fraction_seed=44,
                              max_depth=32, num_leaves=16,
                              min_data_in_leaf =6,subsample = 0.7, min_sum_hessian_in_leaf = 11)

    train_model_and_pred(model, X_train, y_train)


def Cat_Boost():
    from sklearn.metrics import confusion_matrix ,  classification_report
    from catboost import CatBoostClassifier
    c = CatBoostClassifier(iterations=10,
                             learning_rate=0.01,
                             depth=12,
                             eval_metric='F1',
                             random_seed = 120,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 2,
                             od_wait=100)
    c.fit(X_train,y_train,verbose=True)
    pred=c.predict(X_test)
    f1 = f1_score(y_test, pred, average='macro')
    print("macro_F1 score", f1)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))



#Cat_Boost()

#run_GBM()
tree()
#bag_()