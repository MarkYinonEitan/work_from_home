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
import utils_project, utils
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
    out_data_folder  = "OUT_FOLDER"
    net_str     = "NET_STRING"
    weights_file    = "WEIGHTS_FILE"
    resolution  = RESOLUTION_INP
    vx_size     = VX_SIZE_INP

    print("list_file:",list_file)
    print("vx_folder:",vx_folder)
    print("out_data_folder:",out_data_folder)
    print("net_str:",net_str)
    print("weights_file",weights_file)
    print("resolution:",resolution)
    print("vx_size:",vx_size)

    assert_vx_size_and_resolution(vx_size,resolution)

    pdb_emd_pairs = utils_project.read_list_file(list_file)
    all_test_files=[ ]
    for data_pair in pdb_emd_pairs:
        all_test_files  = all_test_files + dbloader.search_for_database_files(vx_folder,data_pair["emd_file"][:-4])

    for test_file in all_test_files:
        #clear graph
        tf.compat.v1.reset_default_graph()

        test_data = EM_DATA(vx_folder,train_pdbs = [test_file], is_random = False)
        print(test_file, '-LOADED')
        utils.blockPrint()
        gan_em_boxes,gan_labels_dict  = train_vaegan.run_test(test_data, net_string = net_str,vae_gan_file = weights_file)
        utils.enablePrint()
        print(test_file, '-CREATED')
        out_dict =  test_data.train_data_dict
        out_dict["data" ] = gan_labels_dict
        out_dict["boxes"] = gan_em_boxes
        dbloader.save_data_dictionary(out_dict, test_file, out_data_folder)
        print(test_file, '-SAVED')
        ed ={}
        dbloader.load_train_data_to_dict([test_file], out_data_folder, ed)
