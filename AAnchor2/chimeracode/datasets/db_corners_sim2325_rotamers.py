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
    input_pdb_folder = data_folder+'/rotamersdata/pdbs_H_added/'
    mrc_maps_folder = data_folder+'/rotamersdata/mrcs/MRCs_23_to_25/'
    target_folder =  data_folder+'/rotamersdata/DB2325simcorners/'
    list_file_name = data_folder+'/rotamersdata/DatasetForBBDepRL2010.txt'



    dbc = DBcreator( input_pdb_folder = input_pdb_folder,
                            mrc_maps_folder = mrc_maps_folder,
                            target_folder = target_folder,
                            file_name_prefix = 'DB_from_',
                            apix = 1.0,
                            label=LabelbyAAType,
                            box_center=BoxCenterAtCG,
                            normalization =Mean0Sig1Normalization,
                            list_file_name = list_file_name,
                            cubic_box_size = 11,
                            is_corners = True,
                            use_list = True)


    runCommand('close all')

    pdbs_sorted = dbc.rotamers_by_pdb_dict.keys()
    pdbs_sorted.sort()
    map_pdb_all = []

    for pdb_id in pdbs_sorted:
        pdb_file_names = glob1(input_pdb_folder,pdb_id+'*.pdb')
        mrc_file_names = glob1(mrc_maps_folder,pdb_id+'*.mrc')

        if ((len(pdb_file_names)==1) and (len(mrc_file_names)==1)):
            map_pdb_all.append((mrc_file_names[0],pdb_file_names[0]))

    for k in range(n,m):
        if k>=len(map_pdb_all):
            "DEBUG 444"
            return
        print "DEBUG 123", map_pdb_all[k][0],map_pdb_all[k][1]
        dbc.create_class_db_corners(map_pdb_all[k][0],map_pdb_all[k][1])

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
