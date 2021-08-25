#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from statistics import mean 
import scipy.stats as stats
from P2P_CONSTANTS import *

#plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (15,14)
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def group_rare_labels(df, var):
	# function takes a dataframe (df) and
    # the variable of interest as arguments

    total_devices = len(df)

    # first I calculate the % of packets for each device
    temp_df = pd.Series(df[var].value_counts() / total_devices)

    # now I create a dictionary to replace the rare labels with the
    # string 'rare' if they are present in less than 5% of devices

    grouping_dict = {
        k: ('rare' if k not in temp_df[temp_df >= 0.05].index else k)
        for k in temp_df.index
    }

    # now I replace the rare categories
    tmp = df[var].map(grouping_dict)

    return tmp

# function to find upper and lower boundaries
# for skewed distributed variables


def find_skewed_boundaries(df, variable, distance):

    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

def diagnostic_plots(df, variable, file_name):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))
	
    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')
	
    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')
	
    plt.savefig(r'{}/{}'.format(PACKETGRAPHICS, file_name), bbox_inches='tight')


def plotBarh(df,var,label,file_name):
	# Plot Frequency graph
	# function takes a dataframe, the variable of interest,
	# plot xlabel and file name as arguments

	df.plot('device_name',var, kind ='barh')
	plt.xlabel(label, fontsize = 20)
	plt.ylabel('IoT Devices',fontsize = 20)
	plt.xscale("log")
	plt.yticks(fontsize=16)
	plt.grid(True)
	plt.savefig(r'{}/{}'.format(PACKETGRAPHICS, file_name), bbox_inches='tight')


def plotLines(df_total,var,label,file_name):
	# Plot Line of Feature Traffic behavior 
	# function takes a dataframe, the variable of interest,
	# plot xlabel and file name as arguments

	sns.set_theme(style="ticks",font='Times New Roman')
	palette = sns.color_palette("hls", n_colors	= 20)
	df_total = df_total.sort_values(['type']).reset_index(drop=True)
	plot = sns.relplot(
			data=df_total,
			x="day_index", y=var, err_style="bars",
			hue="device_name", col="type", style="device_name", kind="line", palette=palette,
			height=5, markers=True, aspect=.75, facet_kws=dict(sharex=False))
	legend = plot.legend.set_title("Device Name")
	(plot.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
	.set_axis_labels("Trace Capture Day", label)
	.set_titles("{col_name}")
	#.set(ylim=(0, 1))
	.tight_layout(w_pad=0))

	plt.yscale("log")
	plt.savefig(r'{}/{}'.format(PACKETGRAPHICS, file_name), bbox_inches='tight')


# get csv with network features 
df_total = pd.read_csv(ALL_PACKETOUTFILE) 

#-------------------
# see data distribution
# missing values handling: receive the median if the distribution is assymetric, and if the distribution is normal (gaussian) they receive the mean
diagnostic_plots(df_total, 'mean_timestamp', 'distribution_mean_timestamp.png')


# looking for outliers,
# using the interquantile proximity rule
# IQR * 1.5, the standard metric
upper_boundary, lower_boundary = find_skewed_boundaries(df_total, 'n_packets', 1.5)
print(upper_boundary)
print(lower_boundary)

#-------------------
# for histograms plots (sum the number of packets and bytes for all days)
df_barh = df_total.groupby(["device_name"]).agg({"n_packets" : np.sum, "sum_n_bytes" : np.sum},axis=1).reset_index()
df_barh = df_barh.rename(columns = {'n_packets' : "Total Packets", 'sum_n_bytes' : "Total Bytes"})
#df_barh = df_barh[df_barh.device_name.isin(['BlipcareBP1','LightBulb1'])] 

var = ['Total Packets','Total Bytes']
#plotBarh(df_barh,var,'Total of Packets and Bytes (Log)','pck-level-total-pck-byte-log.pdf')


# group rare devices
'''
df_total['Devices_Rare_grouped'] = group_rare_labels(df_total, 'n_packets')
df_rare = df_total[df_total.Devices_Rare_grouped.isin(['rare'])]
print(df_rare.head(10))
var = ['n_packets']
plotBarh(df_rare,var,'Total of Packets (Log)','pck-level-total-pck-byte-log-RARE.pdf')
'''

#-------------------
# for line plots
df_total["type"]=0
df_total["type"][df_total.device_name.isin(str(A_IOT_SMALL))] = 'IoT Devices w/ Less No.Packets'
df_total["type"][df_total.device_name.isin(str(B_IOT_BIG))] = "IoT Devices w/ more No.Packets"
df_total["type"][df_total.device_name.isin(str(C_NON_IOT))] = 'Non IoT Devices'
df_total.drop(df_total.loc[df_total['type']==0].index, inplace=True)

#USE => file name = pck-level-numPackets-log.pdf | pck-level-numBytes-log.pdf | pck-level-avgTimestamp-log.pdf
#plotLines(df_total,'n_packets', "Total Number of Packets", "pck-level-numPackets-log.pdf")

