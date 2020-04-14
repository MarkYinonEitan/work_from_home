try:
  import aanchor_all_nets
  import tensorflow as tf
  import gan_net_3d_5
  import gan_net_3d_mean_sigma
except ImportError:
  print("run without TENSORFLOW")

import matplotlib.pyplot as plt
import dbloader
import numpy as np

from dbloader import BATCH_SIZE

def assert_vx_size_and_resolution(vx_size,res):
    if (vx_size != dbloader.VOX_SIZE ) or (res != dbloader.RESOLUTION ):
        raise Exception("VX_SIZE or RES uncorrect")


def read_list_file(list_file):
    pairs=[]
    with open(list_file) as fp:
        line = fp.readline()#read header
        line = fp.readline()
        while line:
            wrds = line.split()
            pdb_id = wrds[0]
            emd_id = wrds[1]
            res = float(wrds[2])
            train_test = wrds[3]
            is_virus = wrds[4]
            map_source = wrds[5]

            line = fp.readline()
            pairs.append({"pdb_file":pdb_id,"emd_file":emd_id,"res":res,"train_test":train_test, "is_virus":is_virus, \
            "map_source":map_source})
    return pairs

def read_thr_file(thr_file):

    thr_dict = {}
    with open(thr_file) as fp:
        line = fp.readline()
        while line:
            wrds = line.split()
            res_type = wrds[0]
            thr = float(wrds[1])

            line = fp.readline()
            thr_dict[res_type] = thr

    return thr_dict

def get_net_by_string(net_string):
    if net_string == 'V5_no_reg':
        return aanchor_all_nets.V5_no_reg()
    if net_string == 'V5_DROP_REG':
        return aanchor_all_nets.V5_Drop_Reg()
    if net_string == 'V5_DROP_REG_2':
        return aanchor_all_nets.V5_Drop_Reg_2()
    if net_string == 'V5_REG_3':
        return aanchor_all_nets.V5_Reg_3()
    if net_string == 'disc_v1':
        return gan_net_3d_5.DISC_V1()
    if net_string == 'gan_v1':
        return gan_net_3d_5.VAE_GAN1()
    if net_string == 'gan_mean_sigma':
        return gan_net_3d_mean_sigma.VAE_GAN1()


    raise Exception('No Net Found')
