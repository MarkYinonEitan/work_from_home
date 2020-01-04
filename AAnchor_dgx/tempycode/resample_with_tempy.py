import os
import sys
import time
import numpy as np
import random
import TEMPy
from  TEMPy import MapParser
from glob import glob1
from scipy.signal import resample


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../../utils/'
ch_path = dir_path + '/../'
sys.path.append(utils_path)
sys.path.append(ch_path)


def resample_by_apix(init_map, new_apix):
    """

    Resample the map based on new_apix sampling.

    Arguments:
        *new_apix*
            Angstrom per pixel sampling

    Return:
        new Map instance

    """
    new_map = init_map.copy()
    apix_ratio = init_map.apix/new_apix
    #new_map.apix = new_apix
    print "DEBUG" , init_map.x_size(),init_map.y_size(),init_map.z_size()
    new_map.fullMap = resample(new_map.fullMap,int( np.ceil(init_map.z_size()*apix_ratio)), axis=0)
    new_map.fullMap = resample(new_map.fullMap, int(np.ceil(init_map.y_size()*apix_ratio)), axis=1)
    new_map.fullMap = resample(new_map.fullMap, int(np.ceil(init_map.x_size()*apix_ratio)), axis=2)
    new_map.apix = (init_map.apix*init_map.box_size()[2])/new_map.box_size()[2]
    return new_map

data_folder = dir_path + '/../data/cryoEM/raw_data/res2729/'
maps_res = []
maps_res.append({"map":'emd-8761.map',"res":2.71})
maps_res.append({"map":'emd-8314.map',"res":2.89})
maps_res.append({"map":'emd-6374.map',"res":2.9})

for map_file in maps_res:
    mrc_file = map_file["map"][:-4]+'.mrc'
    init_map = MapParser.MapParser.readMRC(data_folder+map_file["map"])
    new_apix = map_file["res"]/2.5
    new_map = resample_by_apix(init_map,new_apix)
    new_map.write_to_MRC_file(data_folder+mrc_file)
