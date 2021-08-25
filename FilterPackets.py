# usage: python FilterPackets.py

# import global constants
from P2P_CONSTANTS import *
from FilterPacketsHelper import *
import multiprocessing as MP
import subprocess
import pandas as pd
import os
from tqdm import tqdm

# execute a shell command as a child process
def executeCommand(command, outfilename, rotulos):
    sem.acquire()

    subprocess.call(command, shell=True, stdout=subprocess.PIPE)

    df = pd.read_csv(outfilename, error_bad_lines=False, warn_bad_lines=False)
    # print(df.shape[1])
    df.columns = PACKET_COLUMNS

    df.fillna(0, inplace=True)

    # delete packets with size equal to 0
    df['len'] = df['len_tcp'] + df['len_udp']
    df = df[df['len'] !=0]

    df['src_port'] = df['tcp_srcport'] + df['udp_srcport']
    df['dst_port'] = df['tcp_dstport'] + df['udp_dstport']
    
    df = df.drop(columns=['len_tcp', 'len_udp', 'tcp_srcport', 'tcp_dstport', 'udp_srcport', 'udp_dstport'])

    # No broadcast
    df.drop(df[(df['mac_dst'] == 'ff:ff:ff:ff:ff:ff') | (df['ip_dst'] == '255.255.255.255')].index, inplace=True)

    # add o rotulo do device por meio do mac (usa arq csv)
    df['device_src_name'] = df['mac_src'].map(rotulos.set_index('mac')['device_name'])
    df['device_dst_name'] = df['mac_dst'].map(rotulos.set_index('mac')['device_name'])

    df.to_csv(outfilename, index=False)
    print('done writing packet to : ' + outfilename)
    print(df.values)

    sem.release()


print('----------------------------------------------------')
print('Start Filter Packets\n')

# obtain input parameters and pcapfilenames
inputfiles = getPCapFileNames()
tsharkOptions = getTsharkOptions()
#rotulos = pd.read_csv(ROTULOS)
rotulos_true = pd.read_csv(ROTULOS)
rotulos_fake = pd.read_csv(ROTULOS_FAKE)
rotulos = pd.concat([rotulos_fake, rotulos_true], axis=0, sort=False, ignore_index=True)
print(rotulos)

# create a semaphore so as not to exceed threadlimit
sem = MP.Semaphore(THREADLIMIT)

# get tshark commands to be executed
tasks = []
for filename in inputfiles:
    (command, outfilename) = contructTsharkCommand(filename, tsharkOptions)
    task = MP.Process(target=executeCommand, args=(command, outfilename, rotulos))
    task.start()
    tasks.append(task)

for task in tqdm(tasks, unit="pcap_file"):
    task.join()

print('End Filter Packets')
print('\n--------------------------------------------------')
