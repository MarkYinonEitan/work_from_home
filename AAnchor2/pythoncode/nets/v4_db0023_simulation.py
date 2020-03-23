import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import threading
import timeit
from keras.utils import to_categorical



if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)

import networkanalyser
from networkanalyser import NetworkAnalyser
import resultsplots
from resultsplots import SingleNetResults
import dbloader
import all_nets
from dbloader import LabelbyAAType
#parameters
batch_span=range(200,1000,20)
mini_batch_size = 600
N_valid = 50
N_epoch = 20
mini_epochs = 5
pdbs_per_mini_epoch=200
data_folder = dir_path +'/../../data/'
res_folder = data_folder + '/nets_data/v4_db2325simcorners/'
max_valid_res = 20000
DATABASE_FOLDER =  data_folder +'/rotamersdata/DB2325simcorners/'
WEIGHTS_INITIAL = data_folder +'nets_data/v4_db2325simcorners/weights_updated.h5'

network = all_nets.V5_no_reg()


all_pdbs = glob.glob(DATABASE_FOLDER+'*.pkl.gz')

valid_pdbs = all_pdbs[-N_valid:]
train_pdbs = all_pdbs[:-N_valid]

na =NetworkAnalyser(network, res_folder=res_folder, train_files=train_pdbs,valid_files=valid_pdbs,initial_weight_file=WEIGHTS_INITIAL,mini_batch_size=mini_batch_size,max_valid_res = 20000,labeling = LabelbyAAType)

na.train_network(N_epoch = N_epoch, mini_epochs=mini_epochs,pdbs_per_mini_epoch=pdbs_per_mini_epoch)
