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
import numpy as np
from timeit import default_timer as timer
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import datetime
from tqdm import tqdm
import os.path
import warnings
from P2P_CONSTANTS import *

warnings.filterwarnings('ignore')

def stats(df):
	array = df.values

	# X são as características; Y são as labels
	#Y = df['device_name']	
	df = df.dropna()
	X = df.drop(['device_name'], axis = 1) # seleciona features e ignora a primeira coluna do csv
	Y = df['device_name']  # seleciona rotulos "primeira linha do csv"

	validation_size = 0.40
	seed = 7
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X, Y, test_size=validation_size)
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split (X_train, Y_train, test_size=0.5)

	# Test options and evaluation metric
	scoring = 'accuracy'

	models = []
	#models.append(('NB', GaussianNB()))
	#models.append(('KNN', KNeighborsClassifier())) 
	#models.append(('SVM', SVC(gamma='auto')))	
	#models.append(('AdaBoost', AdaBoostClassifier(n_estimators=50, learning_rate=1))) # OK

	
	# Models used in my analysis
	#models.append(('CART', DecisionTreeClassifier()))
	#models.append(('Kmeans', KMeans(5, random_state=0)))
	#models.append(('R.Forest', RandomForestClassifier(n_estimators=100))) # OK
	#models.append(('Bagging', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed))) # OK
	models.append(('XGBoost',XGBClassifier(n_estimators=50)))


	results, names, f1Results, recallResults, precisionResults, timeResults = [], [], [], [], [], []
	for name, model in tqdm(models, unit = "models") :
		startTemp = timer()
		names.append(name)
		model.fit(X_train, Y_train)
		predictions = model.predict(X_validation)
		g = model.score(X_validation, Y_validation)
		results.append(g)
		accuracy = accuracy_score(Y_validation, predictions)
		f1 = f1_score(Y_validation, predictions, average="macro")
		f1Results.append(f1)
		recall = recall_score(Y_validation, predictions, average="macro")
		recallResults.append(recall)
		precision = precision_score(Y_validation, predictions, average="macro")
		precisionResults.append(precision)
		endTemp = timer()
		#msg = "\n%s: mean: %f tempo: %f" % (name, g.mean(), endTemp-startTemp)
		#print(g)
		msg = "\n%s: accuracy: %f, f1: %f, recall: %f, precision: %f, tempo: %f" % (name, accuracy, f1, recall, precision, endTemp-startTemp)
		print(msg)
		timeResults.append(endTemp-startTemp) 
		#print >> arquivo, msg
		cm = confusion_matrix(Y_validation, predictions)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		#print(cm.diagonal())
	return (names, results, f1Results, recallResults, precisionResults, timeResults)

start = timer()

df = pd.read_csv(ALL_PACKETOUTFILE, sep=",", header='infer')
df = df.drop(columns=['day_index', 'day_capture'])
#print(df.columns)

colorX=['#0000ff','#1919ff', '#3232ff', '#4c4cff', '#6666ff', '#7f7fff', '#9999ff', '#b2b2ff']

result = Parallel(n_jobs=1)(delayed(stats)(df) for i in range(2))

name = []
accuracy_res =[]
f1_score_res =[]
precision_res = []
recall_res = []
tempo_res = []

samples = ""

for resultado in tqdm(result, unit = "resultado"):
	all_devices_samples = []

	name.append(resultado[0][0])
	accuracy_res.append(resultado[1][0])
	f1_score_res.append(resultado[2][0])
	recall_res.append(resultado[3][0])
	precision_res.append(resultado[4][0])
	tempo_res.append(resultado[5][0])

	samples= {
	'name_Classifer':name, 'accuracy' : accuracy_res,
	'F1-score' : f1_score_res, 'recall' : recall_res, 
	'precision' : precision_res, 'tempo' : tempo_res  	
	}

	device_sample = pd.DataFrame(data=samples) 
	all_devices_samples.append(device_sample)

agrupado = pd.concat(all_devices_samples)

# use 'packet-level-MLresults.csv' OR 'flow-level-MLresults.csv'

if os.path.isfile(GRAPHICSDIR+"packet-level-MLresults.csv"):
	fram = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults.csv"), index_col	= 0)
	final = fram.append(agrupado)
	final.to_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults.csv"))
	print(final)
	
else: 
	agrupado.to_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults.csv"))


# Normalize features using MinMaxScalerx for the plots 
#df = df_total.select_dtypes(exclude=['string','object']).columns
#df_total[df] = MinMaxScaler().fit_transform(df_total[df]).round(2)
#df_total.to_csv(r'{}/{}'.format('./', 'out_plot.csv'), header = True, index = False)	