#!/bin/bash
sudo apt-get update
sudo apt-get install python3-tqdm python3-numpy python3-scipy python3-matplotlib ipython3 python3-pandas python3-sympy python3-nose
sudo apt-get install tshark
pip3 install -U scikit-learn 
pip3 install communityid
python3 verifyLibraries.py
