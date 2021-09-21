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
plt.rcParams["font.family"] = ['Times New Roman']


#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow-level-MLresults.csv"), index_col = 0) 

#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyKeyFeats.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow-level-MLresults-onlyKeyFeats.csv"), index_col = 0) 

#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet600fake-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet1000fake-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet5000fake-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet1000_10000fake-level-MLresults.csv"), index_col = 0) 

#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow600fake-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow1000fake-level-MLresults.csv"), index_col = 0) 
#df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow5000fake-level-MLresults.csv"), index_col = 0) 
df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"flow1000_10000fake-level-MLresults.csv"), index_col = 0) 

#F1-score,accuracy,name_Classifer,precision,recall,tempo
df1 = df1.drop(columns=['tempo'])

f = plt.figure()

barwidth=0.20

# posições de cada barra
r1 = np.arange(len(df1))
r2 = [x + barwidth for 	x in r1]
r3 = [x + barwidth for 	x in r2]
r4 = [x + barwidth for 	x in r3]

plt.bar(r1, df1['accuracy'], color='tab:blue', width=barwidth, label='Accuracy')
plt.bar(r2, df1['precision'], color='tab:orange', width=barwidth, label='Precision')
plt.bar(r3, df1['recall'], color='tab:green', width=barwidth, label='Recall')
plt.bar(r4, df1['F1-score'], color='tab:red', width=barwidth, label='F1-Score')

plt.xlabel('ML Algortithms')
plt.xticks([r + barwidth for r in range(4)], df1['name_Classifer'])
#['RF\nw/ IAT', 'RF\nw/o IAT', 'XGBoost\nw/ IAT', 'XGBoost\nw/o IAT'])

plt.ylabel('%')

plt.grid(True, axis='y')


#f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
#          ncol=2, fancybox=True, shadow=True)
plt.legend(loc='lower center', ncol=2)#, fontsize=30)
plt.ylim(0,1)

#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow-level-MLresults-bar")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-bar")), bbox_inches='tight')

#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyKeyFeats")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow-level-MLresults-onlyKeyFeats")), bbox_inches='tight')

#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet600fake-level-MLresults")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet1000fake-level-MLresults")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet5000fake-level-MLresults")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet1000_10000fake-level-MLresults")), bbox_inches='tight')

#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow600fake-level-MLresults")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow1000fake-level-MLresults")), bbox_inches='tight')
#f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow5000fake-level-MLresults")), bbox_inches='tight')
f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"flow1000_10000fake-level-MLresults")), bbox_inches='tight')

