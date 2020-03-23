import os
import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import threading
import timeit
from keras.utils import to_categorical
from glob import glob


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
N_valid = 100
N_epoch = 20
mini_epochs = 5
pdbs_per_mini_epoch=2000

data_folder = dir_path +'/../../data/'
NETS_FOLDER = data_folder+'/nets_data/'
RES_FOLDER = NETS_FOLDER+'/v4_db0023_real_rot10/'
DATABASE_FOLDER =  data_folder +'/cryoEM/DB0023class_rot10/'
WEIGHTS_INITIAL = RES_FOLDER+'/weights_updated.h5'
LABELELING = LabelbyAAType

train_emds = ['8762','8194','3295']
valid_emds = ['2984']

NETWORK = all_nets.V5_no_reg()



train_data_files = []
for train_num in train_emds:
    train_data_files = train_data_files+glob(DATABASE_FOLDER+'*{}*.pkl.gz'.format(train_num))

for t in train_data_files:
    print "DEBUG", t


valid_data_files = []
for valid_num in valid_emds:
    valid_data_files = valid_data_files+glob(DATABASE_FOLDER+'*{}*.pkl.gz'.format(valid_num))

na =NetworkAnalyser(NETWORK, res_folder=RES_FOLDER, train_files=train_data_files,valid_files=valid_data_files,initial_weight_file=WEIGHTS_INITIAL,mini_batch_size=mini_batch_size,labeling = LABELELING)

na.train_network(N_epoch = N_epoch, mini_epochs=mini_epochs,pdbs_per_mini_epoch=pdbs_per_mini_epoch)
