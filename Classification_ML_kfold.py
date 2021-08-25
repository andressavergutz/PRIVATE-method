
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
from timeit import default_timer as timer
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from tqdm import tqdm
import os.path
import warnings
from P2P_CONSTANTS import *

warnings.filterwarnings('ignore')

def stats(df):
    # Split-out validation dataset

    X = df.drop(['device_name'], axis=1)
    Y = df['device_name']  

    validation_size = 0.40
    seed = 7

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed, shuffle=False)
    target_names = df['device_name'].unique()

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    models = []
    # models.append(('NB', GaussianNB()))
    # models.append(('AdaBoost', AdaBoostClassifier(n_estimators=50, learning_rate=1)))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('SVM', SVC(gamma='auto')))
    # models.append(('Kmeans', KMeans(5, random_state=0)))
    
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('R.Forest', RandomForestClassifier(n_estimators=100)))
    # models.append(('Bagging', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed)))
    models.append(('XGBoost',XGBClassifier(n_estimators=50)))

    # Evaluate each model in turn

    results, names, f1Results, recallResults, precisionResults, timeResults = [], [], [], [], [], []
    for name, model in models:
        startTemp = timer()
        kfold = model_selection.KFold(n_splits=9, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(np.mean(cv_results))
        names.append(name)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        g = model.score(X_validation, Y_validation)
        # results.append(g)
        accuracy = accuracy_score(Y_validation, predictions)
        f1 = f1_score(Y_validation, predictions, average="macro")
        # print f1
        f1Results.append(f1)
        recall = recall_score(Y_validation, predictions, average="macro")
        recallResults.append(recall)
        precision = precision_score(
            Y_validation, predictions, average="macro")
        precisionResults.append(precision)
        endTemp = timer()
        # msg = "\n%s: mean: %f tempo: %f" % (name, g.mean(), endTemp-startTemp)
        # print(g)
        msg = "\n%s: accuracy: %f, f1: %f, recall: %f, precision: %f, tempo: %f" % (
            name, np.mean(cv_results), f1, recall, precision, endTemp-startTemp)
        # print(msg)

        timeResults.append(endTemp-startTemp)
        # print >> arquivo, msg
        cm = confusion_matrix(Y_validation, predictions)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm.diagonal())
        print(classification_report(Y_validation, predictions,
                                    target_names=target_names, digits=4))
    return (names, results, f1Results, recallResults, precisionResults, timeResults)


start = timer()

# Load df
df = pd.read_csv(ALL_PACKETOUTFILE, sep=",", header='infer')
df = df.drop(columns=['day_index', 'day_capture'])
# print(df.columns)

colorX = ['#0000ff', '#1919ff', '#3232ff', '#4c4cff',
          '#6666ff', '#7f7fff', '#9999ff', '#b2b2ff']

result = Parallel(n_jobs=1)(delayed(stats)(df) for i in range(9))
#result = Parallel(n_jobs=1)(delayed(stats)(df) for i in range(1))

name = []
accuracy_res = []
f1_score_res = []
precision_res = []
recall_res = []
tempo_res = []

samples = ""

for resultado in tqdm(result, unit="resultado"):
    all_devices_samples = []

    name.append(resultado[0][0])
    accuracy_res.append(resultado[1][0])
    f1_score_res.append(resultado[2][0])
    recall_res.append(resultado[3][0])
    precision_res.append(resultado[4][0])
    tempo_res.append(resultado[5][0])

    samples = {
        'name_Classifer': name, 'accuracy': accuracy_res,
        'F1-score': f1_score_res, 'recall': recall_res,
        'precision': precision_res, 'tempo': tempo_res
    }

    device_sample = pd.DataFrame(data=samples)
    all_devices_samples.append(device_sample)

agrupado = pd.concat(all_devices_samples)

# use 'packet-level-MLresults.csv' OR 'flow-level-MLresults.csv'

if os.path.isfile(GRAPHICSDIR+"packet-level-MLresults-kfold.csv"):
    fram = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,
                       "packet-level-MLresults-kfold.csv"), index_col=0)
    final = fram.append(agrupado)
    final.to_csv(r'{}/{}'.format(GRAPHICSDIR,
                 "packet-level-MLresults-kfold.csv"))
    print(final)

else:
    agrupado.to_csv(r'{}/{}'.format(GRAPHICSDIR,
                    "packet-level-MLresults-kfold.csv"))

# print(size(r0))
# arquivo = open('matrixKfold.txt', 'a')
# print >> arquivo, ('\n------------XXXXXXXXXXXXXXXXX kFold - Flow XXXXXXXXXXXXXXXXX--------------')
# for idx in range(len(r1)):
#	print >> arquivo, ('\n------------XXXXXXXXXXXXXXXXXXXX Run %g XXXXXXXXXXXXXXXXXXXX---------------' % (int(idx)+1))
#	for idx2 in range(len(r1[idx])):
#		idx = int(idx)
#		idx2 = int(idx2)
#		msg = "\n%s: accuracy: %f, f1: %f, recall: %f, precision: %f, tempo: %f" % (r0[idx][idx2], r1[idx][idx2], r2[idx][idx2], r3[idx][idx2], r4[idx][idx2], r5[idx][idx2])
#		print >> arquivo, msg
