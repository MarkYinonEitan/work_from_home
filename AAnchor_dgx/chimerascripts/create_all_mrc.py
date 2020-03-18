import chimera
import os
import sys
import numpy as np
from chimera import runCommand
from glob import glob1
import time
import shutil

utils_path = '/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/pythoncode/utils/'
chimera_path = '/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimeracode/'
sys.path.append(utils_path)
sys.path.append(chimera_path)

from dbcreator import EMmaps
from process_rotamers_data import get_mrc_file_name,get_pdb_id
import utils_project

time_pause=2

def create_mrc_files(inp_list_file,inp_pdb_folder,out_mrc_folder):

    list = utils_project.read_list_file(inp_list_file)

    for data_row in list:
        runCommand('close all')
        time.sleep(time_pause)
        res_map = data_row["res"]
        source_pdb_file = inp_pdb_folder + data_row["pdb_file"]
        target_pdb_file = out_mrc_folder + data_row["pdb_file"]
        target_mrc_file = out_mrc_folder + data_row["emd_file"]

        time.sleep(time_pause)
        #load from pdb
        prot = chimera.openModels.open(source_pdb_file,'PDB')[0]
        #create map
        runCommand('molmap #0 ' + str(res_map) + ' modelId 5')
        time.sleep(time_pause)
        runCommand('volume #5 save ' + target_mrc_file)
        time.sleep(time_pause)
        try:
            shutil.copyfile(source_pdb_file, target_pdb_file)
        except shutil.Error:
            if source_pdb_file==target_pdb_file :
                print("Source and destination represents the same file.")
            else:
                raise shutil.Error()


    return

if __name__ == "chimeraOpenSandbox":
    print "START"
    inp_list_file = sys.argv[3]
    inp_pdb_folder = sys.argv[4]
    out_mrc_folder = sys.argv[5]

    create_mrc_files(inp_list_file,inp_pdb_folder,out_mrc_folder)
    runCommand('stop')
