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



def rotate_maps_and_pdbs(inp_list_file,out_list_file,inp_pdb_folder,inp_mrc_folder,target_folder,N_anlges):

    out_list = []
    list = utils_project.read_list_file(inp_list_file)
    with open(out_list_file,"w") as f:
        f.write("PDB_ID MAP_ID RESOLUTION TRAIN/TEST/VALID/ IS_VIRUS REMARK\n")

    for data_row in list:

        if data_row["is_virus"] == "YES":
            #copy
            # add row to a new list
            target_mrc_file = data_row["emd_file"]
            target_pdb_file = data_row["pdb_file"]

            out_list.append({"pdb_file":target_pdb_file,"emd_file":target_mrc_file,"res":data_row["res"],\
            "train_test":data_row["train_test"], "is_virus":data_row["is_virus"],"remark":"Automatically Created"})

            shutil.copy(inp_mrc_folder + data_row["emd_file"], target_folder)
            shutil.copy(inp_pdb_folder + data_row["pdb_file"], target_folder)

            with open(out_list_file,"a") as f:
                d = out_list[-1]
                f.write(d["pdb_file"]+" " + d["emd_file"]+" " +str(d["res"])+" "+d["train_test"]+\
                " "+d["is_virus"]+" "+d["remark"]+"\n")
            continue

        source_map_file = inp_mrc_folder + data_row["emd_file"]
        source_pdb_file = inp_pdb_folder + data_row["pdb_file"]

        for k in range(N_anlges):
            euler_angles = np.random.rand(3)*30

            target_mrc_file = data_row["emd_file"][:-4] + '_rot{}.mrc'.format(k)
            target_pdb_file = data_row["pdb_file"][:-4]+'_rot{}.pdb'.format(k)

            target_mrc_file_full = target_folder + target_mrc_file
            target_pdb_file_full = target_folder + target_pdb_file

            EMmaps.rotate_map_and_pdb(source_map_file, source_pdb_file, euler_angles, target_mrc_file_full, target_pdb_file_full)
            runCommand('close all')
            out_list.append({"pdb_file":target_pdb_file,"emd_file":target_mrc_file,"res":data_row["res"],\
            "train_test":data_row["train_test"], "is_virus":data_row["is_virus"],"remark":"Automatically Created"})

            with open(out_list_file,"a") as f:
                d = out_list[-1]
                f.write(d["pdb_file"]+" " + d["emd_file"]+" " +str(d["res"])+" "+d["train_test"]+\
                " "+d["is_virus"]+" "+d["remark"]+"\n")



if __name__ == "chimeraOpenSandbox":
    print "START"
    inp_list_file = sys.argv[3]
    out_list_file = sys.argv[4]
    inp_pdb_folder = sys.argv[5]
    inp_mrc_folder = sys.argv[6]
    out_mrc_folder = sys.argv[7]
    N_anlges = np.int(sys.argv[8])


    rotate_maps_and_pdbs(inp_list_file,out_list_file,inp_pdb_folder,inp_mrc_folder,out_mrc_folder,N_anlges)
    runCommand('stop')
