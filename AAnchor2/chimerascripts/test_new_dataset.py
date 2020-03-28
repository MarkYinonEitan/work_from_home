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

import os
cur_pass = os.path.realpath(__file__)

utils_path = cur_pass + '/../pythoncode/utils/'
chimera_path = cur_pass + '/../chimeracode/'
sys.path.append(utils_path)
sys.path.append(chimera_path)

import dbcreator
reload(dbcreator)
import dbloader
reload (dbloader)
from dbcreator import Mean0Sig1Normalization, DBcreator,BoxCenterAtCG
from dbloader import LabelbyAAType,Mean0Sig1Normalization, NoNormalization
import utils_project
import visualization_utils
reload(visualization_utils)

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

dbc.create_class_db_corners(input_mrc_file, input_pdb_file, limits_pdb = limit_boxes ,file_name_suffix = '_0')

f_csv = "/Users/markroza/Documents/work_from_home/data/AAnchor2/first_impl/DB_from_1yti_0"

data_dict = {}
dbloader.load_train_data_to_dict([f_csv],data_dict)


dd_test_corners = filter(lambda x: x['pos']==102, data_dict["data"])
for x in dd_test_corners:
	print(['pos'])
	print(x['CG_pos_X'],x['CG_pos_Y'],x['CG_pos_Z'])
	print(x['box_center_x'],x['box_center_y'],x['box_center_z'])
	print(x["phi"],x["psi"],x['chi1'],x['chi2'],x['chi3'],x['chi4'])

visualization_utils.visual_box_test(f_csv, data_folder+input_pdb_file,)
