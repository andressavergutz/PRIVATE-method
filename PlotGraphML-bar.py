#!/usr/bin/python0

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics 
import sys
from P2P_CONSTANTS import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (15,14)
plt.rcParams['font.size'] = 40
#plt.rcParams["font.family"] = ['Times New Roman']

df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyBytes.csv"), index_col = 0) 

#F1-score,accuracy,name_Classifer,precision,recall,tempo
df1 = df1.drop(columns=['tempo'])
std_by_classifier = df1.groupby(['name_Classifer']).std(ddof=1)
print(std_by_classifier)
mean_by_classifier = df1.groupby(['name_Classifer']).mean()
print(mean_by_classifier)
frames = [mean_by_classifier, std_by_classifier]
by_classifier = pd.concat(frames, axis=1).reset_index()
print(by_classifier)

f = plt.figure()

barwidth=0.20

r1 = np.arange(len(by_classifier))
r2 = [x + barwidth for 	x in r1]
r3 = [x + barwidth for 	x in r2]
r4 = [x + barwidth for 	x in r3]

plt.bar(r1, mean_by_classifier['accuracy'], color='tab:red', width=barwidth, label='Accuracy', yerr=std_by_classifier['accuracy'])
plt.bar(r2, mean_by_classifier['precision'], color='mediumblue', width=barwidth, label='Precision', yerr= std_by_classifier['precision'])
plt.bar(r3, mean_by_classifier['recall'], color='forestgreen', width=barwidth, label='Recall', yerr=std_by_classifier['recall'])
plt.bar(r4, mean_by_classifier['F1-score'], color='tab:orange', width=barwidth, label='F1-Score', yerr=std_by_classifier['F1-score'])

plt.xlabel('ML Algortithms')
plt.xticks([r + barwidth for r in range(4)], by_classifier['name_Classifer'])
#['RF\nw/ IAT', 'RF\nw/o IAT', 'XGBoost\nw/ IAT', 'XGBoost\nw/o IAT'])

plt.ylabel('%')

plt.grid(True, axis='y')


#f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
#          ncol=2, fancybox=True, shadow=True)
plt.legend(loc='lower center', ncol=2)#, fontsize=30)
plt.ylim(0,1)

f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-bar-onlyBytes.pdf")), bbox_inches='tight')
f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-bar-onlyBytes.png")), bbox_inches='tight')

