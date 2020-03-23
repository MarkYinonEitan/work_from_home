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
from dbcreator import Mean0Sig1Normalization, DBcreator,BoxCenterAtCG
from dbloader import LabelbyAAType,Mean0Sig1Normalization, NoNormalization



def create_db_n_m_file(n=0,m=0):
    data_folder = dir_path + '/../../data/'
    input_pdb_folder = data_folder+'/cryoEM/raw_data/res2729_rot10/'
    mrc_maps_folder = data_folder+'/cryoEM/raw_data/res2729_rot10/'
    target_folder =  data_folder+'/cryoEM/DB2729class_rot10/'

    dbc = DBcreator( input_pdb_folder = input_pdb_folder,
                            mrc_maps_folder = mrc_maps_folder,
                            target_folder = target_folder,
                            file_name_prefix = 'DB_from_',
                            apix = 1.0,
                            label=LabelbyAAType,
                            box_center=BoxCenterAtCG,
                            normalization =Mean0Sig1Normalization,
                            list_file_name = None,
                            cubic_box_size = 11,
                            is_corners = True,
                            use_list = False)


    runCommand('close all')
    N = 5


    map_pdb_files_train_virus = []
    map_pdb_files_train_virus.append(("emd-3246.map","5foj_all.pdb",True))
    map_pdb_files_train_virus.append(("emd-8574.map","5uf6_all.pdb",True))
    map_pdb_files_train_virus.append(("emd-7300.map","6bwx.pdb",True))
    map_pdb_files_train_virus.append(("emd-8761.mrc","5w3l_all.pdb",True))
    map_pdb_files_train_virus.append(("emd-8189.map","5k0u_all.pdb",True))
    map_pdb_files_train_virus.append(("emd-8604.map","5us7.pdb",True))
    map_pdb_files_train = []
    map_pdb_files_train.append(("emd-7452.map","6cbe.pdb",False))
    map_pdb_files_train.append(("emd-6741.map","5xnl.pdb",False))


    map_pdb_files_valid = []
    map_pdb_files_valid.append(("emd-6224.map","3j9c_all.pdb",False))

    rot_suffixes = ['_rot{}'.format(x) for x in range(10)]

    map_pdb_box_all = []

    #virus rotation
    for virus_tuple in map_pdb_files_train_virus:
        mrc_file = virus_tuple[0][:-4] + '_rot0.mrc'
        pdb_file = virus_tuple[1][:-4] +  '_rot0.pdb'
        boxes = dbcreator.get_regions(input_pdb_folder+pdb_file,N)
        for box in boxes:
            map_pdb_box_all.append((mrc_file,pdb_file,box))
    #add validation
    for valid_tuple in map_pdb_files_valid:
        mrc_file = valid_tuple[0][:-4] + '_rot0.mrc'
        pdb_file = valid_tuple[1][:-4] +  '_rot0.pdb'
        box = dbcreator.get_regions(input_pdb_folder+pdb_file,1)[0]
        map_pdb_box_all.append((mrc_file,pdb_file,box))
    #add train
    for train_tuple in map_pdb_files_train:
        for rsfx in rot_suffixes:
            mrc_file = train_tuple[0][:-4]+rsfx+'.mrc'
            pdb_file = train_tuple[1][:-4]+rsfx+'.pdb'
            box = dbcreator.get_regions(input_pdb_folder+pdb_file,1)[0]
            map_pdb_box_all.append((mrc_file,pdb_file,box))

    for k in range(n,m):
        if k>=len(map_pdb_box_all):
            "DEBUG 444"
            return
        print "DEBUG 123", map_pdb_box_all[k][0],map_pdb_box_all[k][1]
        dbc.create_class_db_corners(map_pdb_box_all[k][0],map_pdb_box_all[k][1],limits_pdb = map_pdb_box_all[k][2],file_name_suffix = '_'+str(k)+'.pkl.gz' )

if __name__ == "chimeraOpenSandbox":
    k=0
    while sys.argv[k][0:10]!= __file__[0:10]:
        k=k+1
        print sys.argv[k], '###', __file__
    n_file = int(sys.argv[k+1])
    m_file = int(sys.argv[k+2])
    print "RUNINNG FILE NUM",n_file
    create_db_n_m_file(n=n_file,m=m_file)
    runCommand('stop')
