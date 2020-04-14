import os
import sys
import importlib
import numpy as np
import shutil
import tensorflow as tf
from timeit import default_timer as timer



python_path = '/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchorGan3A/code/python/'
sys.path.append(python_path)

import dbloader
from dbloader import EM_DATA
import utils_project
import train_vaegan


def run_discriminator(inp_data_fld, pdb_id, out_file_name,disc_net_str,disc_weights_file):
    em_data = EM_DATA(inp_data_fld,train_pdbs=[pdb_id], is_random = False)
    test_points = utils_project.getTestPoints(em_data)
    res_disc = utils_project.run_disc_on_test_points(test_points, disc_net_str, disc_weights_file)
    out_matrx = utils_project.tstpoint2mtrx(test_points)
    np.save(out_file_name,out_matrx)

def assert_vx_size_and_resolution(vx_size,res):
    if (vx_size != dbloader.VOX_SIZE ) or (res != dbloader.RESOLUTION ):
        raise Exception("VX_SIZE or RES uncorrect")


if __name__ == "__main__":
    list_file   = "LIST_FILE"
    vx_folder   = "VOX_FOLDER"
    out_folder  = "OUT_FOLDER"
    net_str     = "NET_STRING"
    n_epochs    = NUM_EPOCHS
    resolution  = RESOLUTION_INP
    vx_size     = VX_SIZE_INP

    print("list_file:",list_file)
    print("vx_folder:",vx_folder)
    print("out_folder:",out_folder)
    print("net_str:",net_str)
    print("n_epochs:",n_epochs)
    print("resolution:",resolution)
    print("vx_size:",vx_size)

    assert_vx_size_and_resolution(vx_size,resolution)


    pdb_emd_pairs = utils_project.read_list_file(list_file)
    in_train  = list(filter(lambda x: pdb_emd_pairs[x]["train_test"] == "TRAIN", range(len(pdb_emd_pairs))))
    real_train_files = []
    for x in in_train:
        real_train_files = real_train_files + dbloader.search_for_database_files(vx_folder,pdb_emd_pairs[x]["emd_file"][:-4])

    print("DEBUG 2354",real_train_files)

    in_test  = list(filter(lambda x: pdb_emd_pairs[x]["train_test"] == "TEST", range(len(pdb_emd_pairs))))
    real_test_files = []
    for x in in_test:
        real_test_files = real_test_files + dbloader.search_for_database_files(vx_folder,pdb_emd_pairs[x]["emd_file"][:-4])

    real_data_train = EM_DATA(vx_folder,train_pdbs = real_train_files, is_random = True)

    real_data_test = EM_DATA(vx_folder,train_pdbs = real_test_files, is_random = False)

    print("DEBUG 1102",real_data_train.N_batches)
    train_vaegan.run_training(real_data_train,real_data_test, net_string = net_str,out_fld = out_folder)
