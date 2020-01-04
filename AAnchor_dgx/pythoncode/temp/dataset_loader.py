import os
import sys



if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
ch_path = dir_path + '/../../chimeracode/'
sys.path.append(utils_path)
sys.path.append(ch_path)

import dbcreator
reload(dbcreator)
from dbcreator import EMmaps



source_map_file   = dir_path+'/../../data/cryoEM/raw_data/maps/emd-3296.map'
target_pklgz_file = dir_path+'/../../data/temp/det_3296.map'
EMmaps.save_map_positions_as_pklgz(source_map_file,target_pklgz_file)
