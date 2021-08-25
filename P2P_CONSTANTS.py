#!/usr/bin/env python

# columns csv packetdata
PACKET_COLUMNS =  ['ip_src', 'ip_dst', 'proto', 'timestamp', 'len_tcp','len_udp', 'mac_src', 'mac_dst', 'tcp_srcport', 'tcp_dstport', 'udp_srcport', 'udp_dstport']

# columns csv packetplots
PACKET_COLUMNS2 = ['ip_src', 'ip_dst', 'proto', 'timestamp', 'mac_src', 'mac_dst', 'len', 'src_port', 'dst_port', 'device_src_name', 'device_dst_name']

A_IOT_SMALL = ["BlipcareBP1","LightBulb1", "iHome1", "Picturef1", "Smoke1", "Splug1", "Weather1"]
B_IOT_BIG = ["Assistant1","BabyMonitor1","Camera1","Device1", "Motion1", "Sleep1", "Speaker1", "Wswitch1"]
C_NON_IOT = ["Gateway", "Laptop1", "Phone1","Printer1","Tablet1"]

# rotulos columns
COLUMNS_ROTULO = ['device','mac','connection','device_name']

COLORS = ['Grey', 'Purple', 'Blue', 'Green', 'Orange', 'Red']
	
TSHARK_OPT = ["-t e",
			"-T fields",
			"-E separator=,",
			"-e ip.src -e ip.dst -e ip.proto -e frame.time_epoch -e tcp.len -e udp.length -e eth.src -e eth.dst",
			"-e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport",
			'-Y "(ip.proto==6)||(ip.proto==17)"']


ROTULOS="device_list.csv"
ROTULOS_FAKE="device_list_false.csv"

PCAPFILES = ['./pcaps/'] 
PCAPDATADIR = './pcaps/'
PACKETDATADIR = './packetdata/'
ALLSAMPLESDIR = './all_final_samples/'
PACKETPLOTSDIR = './packetplots/'
FLOWPLOTSDIR = './flowplots/'
GRAPHICSDIR = './graphics/'
PACKETGRAPHICS = './graphics/packet-level/'
ALL_PACKETOUTFILE = './all_final_samples/all_packet_samples.csv' 

#control the amount of flows into the same conversasion (or flow)
# necessary to control big flows
FLOWGAP = 10 
SAMPLERANGE = 3

THREADLIMIT = 1
TCP_PROTO = '6'
UDP_PROTO = '17'
UDP_HEADERLENGTH = 8

#utility functions
import os
import sys


#utility functions
import os
def getCSVFiles(dirname):
	csvfiles = []
	for eachfile in os.listdir(dirname):
		if eachfile.endswith('.csv'):
			csvfiles.append(dirname + eachfile)	
	return csvfiles