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
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)

import dbcreator
import dbloader
from dbcreator import  DBcreator
from dbloader import Mean0Sig1Normalization


def create_data_set(map_file, pdb_file,out_file):
    runCommand('close all')
    pdbs_folder = os.path.dirname(pdb_file)
    mrc_maps_folder = os.path.dirname(map_file)
    target_folder = os.path.dirname(out_file)


    pdb_file = os.path.basename(pdb_file)
    map_file = os.path.basename(map_file)
    out_file = os.path.basename(out_file)

    dbc = DBcreator(input_pdb_folder = pdbs_folder, mrc_maps_folder = mrc_maps_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,step_for_detection = 1,nones_ratio = 1000,list_file_name=None)

    print "DEBUG 21", map_file,pdb_file
    dbc.create_det_db_labeled(map_file,pdb_file,file_name = out_file)

if __name__ == "chimeraOpenSandbox":
    k=0
    file_name = os.path.basename(sys.argv[0])
    while file_name[0:10]!= __file__[0:10]:
        k=k+1
        file_name = os.path.basename(sys.argv[k])
    map_file = sys.argv[k+1]
    pdb_file = sys.argv[k+2]
    out_file = sys.argv[k+3]

    create_data_set(map_file, pdb_file,out_file)
    runCommand('stop')
