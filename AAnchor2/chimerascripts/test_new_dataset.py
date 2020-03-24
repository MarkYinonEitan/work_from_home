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



utils_path = '/Users/markroza/Documents/GitHub/work_from_home/AAnchor2/pythoncode/utils/'
chimera_path = '/Users/markroza/Documents/GitHub/work_from_home/AAnchor2/chimeracode/'

sys.path.append(utils_path)
sys.path.append(chimera_path)

import dbcreator
from dbcreator import Mean0Sig1Normalization, DBcreator,BoxCenterAtCG
from dbloader import LabelbyAAType,Mean0Sig1Normalization, NoNormalization
import utils_project


data_folder = "/Users/markroza/Documents/work_from_home/data/AAnchor2/first_impl/"

input_pdb_folder = data_folder
mrc_maps_folder  = data_folder
target_folder    = data_folder
input_pdb_file   =  "1yti.pdb"
input_mrc_file   =   "1yti.mrc"

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

N_divide = 1

limit_boxes      = dbcreator.get_regions(data_folder+input_pdb_file,N_divide)[0]

dbc.create_class_db_corners(input_mrc_file, input_pdb_file, limits_pdb = limit_boxes, file_name_suffix = '_0.pkl' )

f.close()
f = open("/Users/markroza/Documents/work_from_home/data/AAnchor2/first_impl/DB_from_1yti_0.pkl",'r')
dd = pickle.load(f)
dd_test_corners = filter(lambda x: x["ref_data"]['pos']==102, dd)
for x in dd_test_corners:
	print(x["ref_data"]['pos'])
	print(x["ref_data"]['CG_pos'])
	print(x["ref_data"]['box_center'])
	print(x["rot_angles"])
