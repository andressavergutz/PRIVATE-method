from P2P_CONSTANTS import *
import os
#return a list of filenames of pcapfiles taken from InputFiles.txt
#if a directory is found then all *.pcap files in the directory are
#included(non-recursive)

def getPCapFileNames():
	
	lines = PCAPFILES
	
	pcapfilenames = []
	for eachline in lines:
		if eachline.endswith('.pcap'):
			if os.path.exists(eachline):
				pcapfilenames.append(eachline)
			else:
				print(eachline + ' does not exist')
				exit()
		else:
			if os.path.isdir(eachline):
				for eachfile in os.listdir(eachline):
					if eachfile.endswith('.pcap'):
						pcapfilenames.append(eachline.rstrip('/') + '/' + eachfile)
			else:
				print(eachline + ' is not a directory')
				exit()
	return pcapfilenames

#return a list of options to be used with tshark
def getTsharkOptions():
	return TSHARK_OPT

def contructTsharkCommand(filename,tsharkOptions):
	command = 'tshark -r ' + filename + ' '
	for eachstring in tsharkOptions:
		command = command + eachstring + ' '
	
	#construct output filename
	outfilename = filename.split('/')

	#cria csv para cada pcap lido no diretorio dos packetdata
	outfilename = PACKETDATADIR + outfilename[len(outfilename)-1] + '.csv'

	#comando para cada pcap
	command += "> '" +outfilename+"'"
	
	return (command,outfilename)

