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

from dbcreator import Mean0Sig1Normalization, DBcreator




def create_db_from_n_to_m(n=0,m=10**10):
    runCommand('close all')

    data_folder = dir_path + '/../../data/'
    input_folder = data_folder+'/cryoEM/raw_data/res0023_aug10/'
    target_folder = data_folder+'/cryoEM/DB0023classrot10pos10/'

    dbc_train = DBcreator(input_pdb_folder = input_folder, mrc_maps_folder = input_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,rand_position = True)

    dbc_valid = DBcreator(input_pdb_folder = input_folder, mrc_maps_folder = input_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,rand_position = False)


    map_pdb_files_train = []
    map_pdb_files_valid = []
    map_pdb_files_train.append(("emd-8194.map","5k12.pdb"))
    map_pdb_files_train.append(("emd-3295.map","5ftj.pdb"))

    map_pdb_files_valid.append(("emd-2984.map","5a1a.pdb"))

    rot_suffixes = ['_rot{}'.format(x) for x in range(10)]
    pos_suffixes = ['_pos{}.pkl.gz'.format(x) for x in range(10)]

    k_map_pdb__suffix_all = {}
    k=0
    #create dictionary
    for train_tuple in map_pdb_files_train:
        for rsfx in rot_suffixes:
            mrc_file = train_tuple[0][:-4]+rsfx+'.mrc'
            pdb_file = train_tuple[1][:-4]+rsfx+'.pdb'
            for psfx in pos_suffixes:
                k_map_pdb__suffix_all[k] = ((mrc_file,pdb_file,psfx))
                k=k+1
    print "DEBUG k", k-1

    for k in range(n,m+1):
        mrc_file = k_map_pdb__suffix_all[k][0]
        pdb_file = k_map_pdb__suffix_all[k][1]
        sfx      = k_map_pdb__suffix_all[k][2]
        print "DEBUG 32",mrc_file,pdb_file,sfx
        dbc_train.create_classification_db(mrc_file,pdb_file,file_name_suffix = sfx)


if __name__ == "chimeraOpenSandbox":
    k=0
    while sys.argv[k][0:10]!= __file__[0:10]:
        k=k+1
        print sys.argv[k], '###', __file__
    n_start = int(sys.argv[k+1])
    n_end = int(sys.argv[k+2])
    print n_start, n_end
    print "DEBUG 3",
    create_db_from_n_to_m(n=n_start,m=n_end)
    runCommand('stop')
