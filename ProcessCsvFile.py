import numpy as np
import multiprocessing as MP
from P2P_CONSTANTS import *
import pandas as pd
import os
from tqdm import tqdm
from statistics import mean, stdev

def cria_pasta_salva(file_name, folder_path, data_frame, index = False, header = True):
	while True:
		try:
			data_frame.to_csv(r'{}/{}'.format(folder_path, file_name), header = header, index = index)	
			break
		except FileNotFoundError:
			os.system("mkdir {}".format(folder_path))

def statistical_features_by_day_perFlow(df_total, file):

	"""
	Manipulacao dos dados para extracao de medidas estatisticas no nivel de fluxo
	Organizacao de saidas para os plots dos graficos
	Para cada grupo de device extrai um subgrupo
	"""
	print(df_total.columns)
	
	flowGroup = df_total.groupby(['ip_src','ip_dst','src_port','dst_port','proto'])
	
	all_devices_samples = []
	for name, amostra in tqdm(flowGroup, unit = "device"): # tqdm = progress bar 
		if (len(amostra.index) >= 2):

			# use for plots
			SAMPLERANGE = len(amostra)		
			amostras_tamanho_x = np.array_split(amostra, len(amostra)/SAMPLERANGE)
			# use for ML algorithms
			#amostras_tamanho_x = np.array_split(amostra, len(amostra)/5)

			for device in amostras_tamanho_x:
				device = pd.DataFrame(device)
				device['iat'] = device['timestamp'].diff()
				device['iat'].fillna(device['iat'].mean(), inplace=True)
				#print(device['iat'].values)
				
				all_devices_samples.append(
					pd.DataFrame(data={
					
					"device_name": [str(device["device_src_name"].values[0])],
					"flow": [str(device["ip_src"].values[0]+str(device["ip_dst"].values[0]))],
					"n_packets":  [device['device_src_name'].count()], 
					"stdev_n_bytes": [device['len'].std(ddof=0)],
					"min_n_bytes": [device['len'].min()],
					"max_n_bytes": [device['len'].max()],
					"sum_n_bytes": [device['len'].sum()],
					"median_n_bytes": [device['len'].median()],
					"mean_timestamp": [device['timestamp'].mean()],
					"stdev_timestamp": [device['timestamp'].std(ddof=0)],
					"sum_timestamp": [device['timestamp'].sum()],
					"median_timestamp": [device['timestamp'].median()],
					"min_timestamp": [device['timestamp'].min()],
					"max_timestamp": [device['timestamp'].max()],
					"mean_iat": [device['iat'].mean()], 
					"stdev_iat": [device['iat'].std(ddof=0)],
					"min_iat": [device['iat'].min()],
					"max_iat": [device['iat'].max()],
					"sum_iat": [device['iat'].sum()],
					"median_iat": [device['iat'].median()],
					"start_timestamp": [device['timestamp'].loc[device.first_valid_index()]],
					"end_timestamp": [device['timestamp'].loc[device.last_valid_index()]],
					"flow_duration": [device['timestamp'].loc[device.last_valid_index()] - device['timestamp'].loc[device.first_valid_index()]]})) 
					
	all_devices_samples = pd.concat(all_devices_samples)
	
	all_devices_samples.to_csv(FLOWPLOTSDIR + (file.split('/')[-1]), index=False)
	print('done writing statistical features to : ' + FLOWPLOTSDIR + (file.split('/')[-1]))

	return all_devices_samples


def statistical_features_by_day_perPacket(df_total, file):

	"""
	Manipulacao dos dados para extracao de medidas estatisticas no nivel de pacote
	Organizacao de saidas para os plots dos graficos
	Para cada grupo de device extrai um subgrupo
	"""
	print(df_total.columns)
	devicegroup = df_total.groupby(['device_src_name'])
	
	all_devices_samples = []
	for name, amostra in tqdm(devicegroup, unit = "device"): # tqdm = progress bar 
		if (len(amostra.index) >= 5):
		
			# use for plots
			#SAMPLERANGE = len(amostra)		
			#amostras_tamanho_x = np.array_split(amostra, len(amostra)/SAMPLERANGE)
			# use for ML algorithms
			amostras_tamanho_x = np.array_split(amostra, len(amostra)/5)

			for device in amostras_tamanho_x:

				device = pd.DataFrame(device)
				device['iat'] = device['timestamp'].diff()
				device['iat'].fillna(device['iat'].mean(), inplace=True)
				#print(device['iat'].values)
				
				all_devices_samples.append(
					pd.DataFrame(data={
					# ['ip_src', 'ip_dst', 'proto', 'timestamp', 'mac_src', 'mac_dst', 'len', 'src_port', 'dst_port', 'device']
					
					"device_name": [str(device["device_src_name"].values[0])],
					#"n_packets":  [device['device_src_name'].count()], 
					"mean_n_bytes": [device['len'].mean()], 
					"stdev_n_bytes": [device['len'].std(ddof=0)],
					"min_n_bytes": [device['len'].min()],
					"max_n_bytes": [device['len'].max()],
					"sum_n_bytes": [device['len'].sum()],
					"median_n_bytes": [device['len'].median()],
					"mean_timestamp": [device['timestamp'].mean()],
					"stdev_timestamp": [device['timestamp'].std(ddof=0)],
					"sum_timestamp": [device['timestamp'].sum()],
					"median_timestamp": [device['timestamp'].median()],
					"min_timestamp": [device['timestamp'].min()],
					"max_timestamp": [device['timestamp'].max()],
					"mean_iat": [device['iat'].mean()], 
					"stdev_iat": [device['iat'].std(ddof=0)],
					"min_iat": [device['iat'].min()],
					"max_iat": [device['iat'].max()],
					"sum_iat": [device['iat'].sum()],
					"median_iat": [device['iat'].median()]})) 
			
	all_devices_samples = pd.concat(all_devices_samples)
	
	all_devices_samples.to_csv(PACKETPLOTSDIR + (file.split('/')[-1]), index=False)
	print('done writing statistical features to : ' + PACKETPLOTSDIR + (file.split('/')[-1]))

	return all_devices_samples


def prepareFinalDF(df):
	# add day column
	df['day_index'] = str(i+1)
	df['day_capture'] = str(file.split('/')[-1].split('.')[-3])

	# add value 0 for devices that not generate traffic
	device_exist = rotulos['device_name'].isin(df['device_name'])
	device_not_exist = rotulos.loc[device_exist != True,['device_name']]
	df = pd.concat([df, device_not_exist], ignore_index=True, axis=0).fillna(0)

	return df

print('----------------------------------------------------')
print("Start to compute Statisitcal Network Traffic Features ")

# get rotulos names
#rotulos = pd.read_csv(ROTULOS)
rotulos_true = pd.read_csv(ROTULOS)
rotulos_fake = pd.read_csv(ROTULOS_FAKE)
rotulos = pd.concat([rotulos_fake, rotulos_true], axis=0, ignore_index=True)
csvfiles = sorted(getCSVFiles(PACKETDATADIR)) # ordena lista

df_flow = pd.DataFrame()
df_packet = pd.DataFrame()
for i, file in tqdm(enumerate(csvfiles), unit = "csv_file"):
	
	atual = pd.read_csv(file)
	if atual.empty:
		print("Warning >> CSV empty : " + file)	

	allFeatures_Packet = statistical_features_by_day_perPacket(atual, file)
	allFeatures_Flow = statistical_features_by_day_perFlow(atual, file)

	allFeatures_Packet = prepareFinalDF(allFeatures_Packet)
	allFeatures_Flow = prepareFinalDF(allFeatures_Flow)

	df_packet = pd.concat([df_packet, allFeatures_Packet], axis=0, sort=False, ignore_index=True)
	df_flow = pd.concat([df_flow, allFeatures_Flow], axis=0, sort=False, ignore_index=True)
	

# file for graphics/plots
cria_pasta_salva("all_packet_samples.csv", ALLSAMPLESDIR, df_packet)
cria_pasta_salva("all_flow_samples.csv", ALLSAMPLESDIR, df_flow)

print("End of ProcessCSVFile")
print('----------------------------------------------------')

