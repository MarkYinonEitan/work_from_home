import os
import sys
import time
import numpy as np
import random
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../../utils/'
ch_path = dir_path + '/../'
sys.path.append(utils_path)
sys.path.append(ch_path)

import dbcreator
import dbloader
from dbcreator import  DBcreator
from dbloader import Mean0Sig1Normalization


runCommand('close all')



data_folder = dir_path + '/../../data/'
mrc_maps_folder = data_folder+'/cryoEM/raw_data/res0023/'
pdbs_folder = data_folder+'/cryoEM/raw_data/res0023/'
target_folder = data_folder+'/temp/'
dbc = DBcreator(input_pdb_folder = pdbs_folder, mrc_maps_folder = mrc_maps_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,step_for_detection = 1,nones_ratio = 1000)

map_file = "emd-2984.map"
pdb_file = "5a1a.pdb"
dbc.create_det_db_labeled(map_file,pdb_file,file_name = 'det2984.pkl.gz')
