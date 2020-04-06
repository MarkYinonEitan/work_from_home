import os
import sys
import importlib
import numpy as np
import shutil
import tensorflow as tf
from timeit import default_timer as timer



python_path = '//specific//netapp5_2//iscb//wolfson/Mark//git//work_from_home/AAcryoGAN3/code/python'
sys.path.append(python_path)

import dataset_loader
importlib.reload(dataset_loader)
from dataset_loader import EM_DATA
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
import net_3d_5
import utils_project
import train_vaegan




if __name__ == "__main__":

    print("Marik")


    list_file   = sys.argv[1]
    # vx_folder   = sys.argv[2]
    # out_folder  = sys.argv[3]
    # net_str     = sys.argv[4]
    # n_epochs    = np.float(sys.argv[5])
    # resolution  = np.float(sys.argv[6])
    # vx_size     = np.float(sys.argv[7])
    # #
    print("list_file:",list_file)
    # print("vx_folder:",vx_folder)
    # print("out_folder:",out_folder)
    # print("net_str:",net_str)
    # print("n_epochs:",n_epochs)
    # print("resolution:",resolution)
    # print("vx_size:",vx_size)
    # #
    # #assert_vx_size_and_resolution(vx_size,resolution)
    # #real_data_train, real_data_test = train_vaegan.get_data_for_train(vx_fold = vx_folder,list_file = list_file)
    # # print("DEBUG 1102",real_data_train.N_batches)
    # # train_vaegan.run_training(real_data_train,real_data_test, net_string = net_str,out_fld = out_folder)
