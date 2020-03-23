import chimera
import os
import sys
import numpy as np
from chimera import runCommand
from glob import glob1
import random
import time
#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../pythoncode/utils/'
sys.path.append(utils_path)

from process_rotamers_data import get_mrc_file_name,get_pdb_id

resolution = [2.9, 3.1]
#folders
input_PDB_folder = dir_path+'/../data/cryoEM/raw_data/res2931/'
target_MRC_folder = dir_path+'/../data/cryoEM/raw_data/res2931_mrc/'

time_pause = 5.1


existing_pdbs_file_names = glob1(input_PDB_folder,'*.pdb')

problematic_pdbs = [];

k=0
for pdb_file in existing_pdbs_file_names:
    #delete
    runCommand('close all')
    time.sleep(time_pause)
    # randomize apix
    res_map = random.uniform(resolution[0],resolution[1])
    apix = random.choice(np.arange(res_map/3.0,res_map/2.0,0.01))
    pdb_name = pdb_file[:-4]
    print (k, pdb_name)
    time.sleep(time_pause)
    #load from pdb
    prot = chimera.openModels.open(input_PDB_folder+pdb_file,'PDB')[0]
    #create map
    runCommand('molmap #0 ' + str(res_map)+' gridSpacing ' + str(apix) + ' modelId 5')
    time.sleep(time_pause)
    #save
    mrc_file_name = get_mrc_file_name(pdb_name,res_map,apix)
    print "DEBUG", mrc_file_name
    runCommand('volume #5 save ' + target_MRC_folder+ mrc_file_name)
    time.sleep(time_pause)

#Exit chimera
runCommand('stop')
