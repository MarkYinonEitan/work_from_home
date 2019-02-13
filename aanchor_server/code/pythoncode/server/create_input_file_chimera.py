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
utils_path = dir_path + '/../utils/'
ch_path = dir_path + '/../../chimeracode/'
sys.path.append(utils_path)
sys.path.append(ch_path)

import dbcreator
from dbcreator import Mean0Sig1Normalization, DBcreator,BoxCenterAtCG
from dbloader import LabelbyAAType,Mean0Sig1Normalization, NoNormalization



def create_candidates_file(source_file,target_file):

    dbc = DBcreator(file_name_prefix = 'DB_from_',
                            apix = 1.0,
                            label=LabelbyAAType,
                            box_center=BoxCenterAtCG,
                            normalization =Mean0Sig1Normalization,
                            cubic_box_size = 11,
                            use_list = False)


    runCommand('close all')



    dbc.create_unlabeled_db(source_file,target_file)

if __name__ == "chimeraOpenSandbox":
    print "START"
    inp_file = sys.argv[3]
    out_file = sys.argv[4]
    create_candidates_file(inp_file,out_file)
    runCommand('stop')
