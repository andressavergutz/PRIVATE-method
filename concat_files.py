import numpy as np
import multiprocessing as MP
from P2P_CONSTANTS import *
import pandas as pd
import os
from tqdm import tqdm


def cria_pasta_salva(file_name, folder_path, data_frame, index = False, header = True):
	while True:
		try:
			data_frame.to_csv(r'{}/{}'.format(folder_path, file_name), header = header, index = index)	
			break
		except FileNotFoundError:
			os.system("mkdir {}".format(folder_path))


def read_samples(file, flow):

	df = pd.read_csv(file)
	index_false = df[ df['day_capture'].eq("21-08-16") ].index
	df.drop(index_false , inplace=True)

	if flow:
		df = df.drop(columns=['flow','day_index', 'day_capture'])
	else: 
		df = df.drop(columns=['day_index', 'day_capture'])

	return df


def concat_files(real_file, fake_file, outfilename,flow):

	fake = pd.read_csv(fake_file)
	print(fake.size)
	
	if flow:
		fake = fake.drop(columns=['flow'])

	fake = pd.concat([fake, real_file], axis=0, sort=False, ignore_index=True)
		
	cria_pasta_salva(outfilename, ALLSAMPLESDIR, fake)

	print('done concating real and fake files to : ' + ALLSAMPLESDIR + (outfilename.split('/')[-1]))
	print(fake.size)


if __name__ == "__main__":


	real_flows = read_samples(ALL_FLOWOUTFILE, True)

	concat_files(real_flows, FAKEFLOW+'06-06-06.pcap.csv', "all_flow_samples_600fake.csv",True)
	concat_files(real_flows, FAKEFLOW+'10-10-10.pcap.csv', "all_flow_samples_1000fake.csv", True)
	concat_files(real_flows, FAKEFLOW+'05-05-05.pcap.csv', "all_flow_samples_5000fake.csv", True)
	concat_files(real_flows, FAKEFLOW+'11-11-11.pcap.csv', "all_flow_samples_1000_10000fake.csv",True)

	real_packets = read_samples(ALL_PACKETOUTFILE, False)

	concat_files(real_packets, FAKEPACKET+'06-06-06.pcap.csv', "all_packet_samples_600fake.csv",False)
	concat_files(real_packets, FAKEPACKET+'10-10-10.pcap.csv', "all_packet_samples_1000fake.csv",False)
	concat_files(real_packets, FAKEPACKET+'05-05-05.pcap.csv', "all_packet_samples_5000fake.csv",False)
	concat_files(real_packets, FAKEPACKET+'11-11-11.pcap.csv', "all_packet_samples_1000_10000fake.csv",False)

