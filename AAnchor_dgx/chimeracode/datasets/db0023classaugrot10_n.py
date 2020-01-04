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
from VolumeViewer import open_volume_file


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../../utils/'
ch_path = dir_path + '/../'
sys.path.append(utils_path)
sys.path.append(ch_path)

import dbcreator
from dbcreator import Mean0Sig1Normalization, DBcreator



def create_db_n_file(n=0):
    runCommand('close all')

    data_folder = dir_path + '/../../data/'
    input_folder = data_folder+'/cryoEM/raw_data/res0023_aug10/'
    target_folder = data_folder+'/cryoEM/DB0023class_rot10/'

    dbc = DBcreator(input_pdb_folder = input_folder, mrc_maps_folder = input_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,is_corners=True)

    map_pdb_files_train_no_rotation = []
    map_pdb_files_train_no_rotation.append(("emd_8762.mrc","5w3m_all.pdb"))
    map_pdb_files_train = []
    map_pdb_files_train.append(("emd-8194.map","5k12.pdb"))
    map_pdb_files_train.append(("emd-3295.map","5ftj.pdb"))
    map_pdb_files_valid = []
    map_pdb_files_valid.append(("emd-2984.map","5a1a.pdb"))

    rot_suffixes = ['_rot{}'.format(x) for x in range(10)]

    map_pdb_all = []
    #add no rotation
    for no_rot_tuple in map_pdb_files_train_no_rotation:
        map_pdb_all.append(no_rot_tuple)
    #add validation
    for valid_tuple in map_pdb_files_valid:
        map_pdb_all.append(valid_tuple)
    #add train
    for train_tuple in map_pdb_files_train:
        for rsfx in rot_suffixes:
            mrc_file = train_tuple[0][:-4]+rsfx+'.mrc'
            pdb_file = train_tuple[1][:-4]+rsfx+'.pdb'
            map_pdb_all.append((mrc_file,pdb_file))

    if n>=len(map_pdb_all):
        "DEBUG 444"
        return
    print "DEBUG 123"
    dbc.create_classification_db(map_pdb_all[n][0],map_pdb_all[n][1])

if __name__ == "chimeraOpenSandbox":
    k=0
    while sys.argv[k][0:10]!= __file__[0:10]:
        k=k+1
        print sys.argv[k], '###', __file__
    n_file = int(sys.argv[k+1])
    print "RUNINNG FILE NUM",n_file
    create_db_n_file(n=n_file)
    runCommand('stop')
