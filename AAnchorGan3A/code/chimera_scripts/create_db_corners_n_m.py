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


cur_pass = os.path.dirname(os.path.realpath(__file__))

utils_path = cur_pass + '/../python/'
chimera_path = cur_pass + '/../chimera/'
sys.path.append(utils_path)
sys.path.append(chimera_path)


import dbcreator
from dbcreator import Mean0Sig1Normalization, DBcreator,BoxCenterAtCG
from dbloader import LabelbyAAType,Mean0Sig1Normalization, NoNormalization
import utils_project



def create_db_n_m_file(input_list_file, input_folder, output_folder, error_list_file,n=0,m=0):
    input_pdb_folder = input_folder
    mrc_maps_folder  = input_folder
    target_folder    = output_folder

    list_data = utils_project.read_list_file(input_list_file)

    dbc = DBcreator( input_pdb_folder = input_pdb_folder,
                            mrc_maps_folder = mrc_maps_folder,
                            target_folder = target_folder,
                            label=LabelbyAAType,
                            box_center=BoxCenterAtCG,
                            normalization =Mean0Sig1Normalization,
                            list_file_name = None,
                            is_corners = True,
                            use_list = False)


    runCommand('close all')

    creation_dicts=[]

    for data_row in list_data:
        input_pdb_file =   data_row["pdb_file"]
        input_mrc_file = data_row["emd_file"]
        if data_row["is_virus"]=="YES":
            N_divide = 5
        else:
            N_divide = 1

        limit_boxes      = dbcreator.get_regions(input_folder + input_pdb_file,N_divide)

        for l_box in limit_boxes:
            creation_dicts.append({"pdb_file":input_pdb_file, "emd_file":input_mrc_file, "limit_box":l_box, "map_source":data_row["map_source"] })



    for k in range(n,m):
        if k>=len(creation_dicts):
            return
        print(n,m,k,creation_dicts[k]["emd_file"],creation_dicts[k]["pdb_file"],creation_dicts[k]["limit_box"])
        try:
            dbc.create_class_db_corners(creation_dicts[k]["emd_file"], creation_dicts[k]["pdb_file"], limits_pdb = creation_dicts[k]["limit_box"], map_source = creation_dicts[k]["map_source"], file_name_suffix = '_'+str(k) )
        except Exception as e:
            with open(error_list_file,"a") as f:
                d = out_list[-1]
                f.write(creation_dicts[k]["emd_file"])
                f.write('\n')
                f.write(creation_dicts[k]["pdb_file"])
                f.write('\n')
                f.write(creation_dicts[k]["limit_box"])
                f.write('\n')
                f.write(sys.exc_info()[0])
                f.write('================\n')


if __name__ == "chimeraOpenSandbox":
    k=0
    while k<7:
        print k, sys.argv[k], '###', __file__
        k=k+1
    k=2
    input_list_file = sys.argv[k+1]
    input_folder = sys.argv[k+2]
    output_folder = sys.argv[k+3]
    error_list_file  = sys.argv[k+4]
    n_file = int(sys.argv[k+5])
    m_file = int(sys.argv[k+6])
    print "RUNINNG FILE NUM",n_file
    create_db_n_m_file(input_list_file,  input_folder, output_folder,error_list_file,n=n_file,m=m_file)
    runCommand('stop')
