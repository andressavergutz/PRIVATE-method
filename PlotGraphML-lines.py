#!/usr/bin/python0

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean 
import sys
from P2P_CONSTANTS import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (15,14)
plt.rcParams['font.size'] = 40
#plt.rcParams["font.family"] = ['Times New Roman']


def window_average(x,N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while(high_index<len(x)):
        temp = sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index = low_index + N
        high_index = high_index + N
    return w_avg

df1 = pd.read_csv(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyBytes.csv"), index_col = 0) 
print(df1)

#file = pd.concat([df1,df2])
#file = pd.concat([df3,df4])
file = df1
#print(df1)
#dataset = sys.argv[1]


by_classifier = file.groupby(['name_Classifer'])
samples = []
for classifyer in by_classifier:
    samples.append({ 'rotulo':classifyer[0], 'sample': classifyer[1]['F1-score'].values.tolist()})
    #print(len(classifyer[1]['accuracy'].values.tolist()))
f = plt.figure()
datas = []

for sample in samples:
    data = np.linspace(0, 30, len(sample['sample']), endpoint=True)
    plt.plot(data, sample['sample'], label=sample['rotulo'], marker="v")
    datas.append( {'rotulo':sample['rotulo'], 'sample':  data})


plt.xlabel('Replication')
plt.ylabel('F1-score(%)')
plt.grid(True, axis='y')
plt.legend(loc='lower left', fontsize=30)
plt.ylim(0,1.1)

## acurracy e F-score 

f.savefig("{}.pdf".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyBytes.pdf")), bbox_inches='tight')
f.savefig("{}.png".format(r'{}/{}'.format(GRAPHICSDIR,"packet-level-MLresults-onlyBytes.png")), bbox_inches='tight')
#f.savefig("{}.png".format(dataset), bbox_inches='tight')
#f.savefig("acuracias{}.eps".format(dataset), bbox_inches='tight')