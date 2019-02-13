#import chimera
import os
import sys
from ftplib import FTP
import dbcreator
from numpy import random
from shutil import copyfile

#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../pythoncode/utils/'
sys.path.append(utils_path)

from dbcreator import EMmaps

SOURCE_DATA_FOLDER = dir_path +'/../data/cryoEM/raw_data/res2931/'
TARGET_DATA_FOLDER = dir_path +'/../data/cryoEM/raw_data/res2931_rot10/'
N = 10
input_files = []
input_files.append({"map":'emd-3014.mrc',"pdb":'5a33_all.pdb','rotate':False})
input_files.append({"map":'emd-3574.mrc',"pdb":'5mv5_all.pdb','rotate':False})
input_files.append({"map":'emd-3631.mrc',"pdb":'5nej_all.pdb','rotate':False})
input_files.append({"map":'emd-3713.mrc',"pdb":'5nwy.pdb','rotate':True})
input_files.append({"map":'emd-3999.mrc',"pdb":'6ezj.pdb','rotate':False})
input_files.append({"map":'emd-6555.mrc',"pdb":'3jci_all.pdb','rotate':False})
input_files.append({"map":'emd-6675.mrc',"pdb":'5wq7.pdb','rotate':True})
input_files.append({"map":'emd-6714.mrc',"pdb":'5xb1.pdb','rotate':True})
input_files.append({"map":'emd-3842.mrc',"pdb":'5ool.pdb','rotate':True})
input_files.append({"map":'emd-7436.mrc',"pdb":'6c9i.pdb','rotate':True})
input_files.append({"map":'emd-7526.mrc',"pdb":'6cmx.pdb','rotate':True})

#validation
input_files.append({"map":'emd-7050.mrc',"pdb":'6b46.pdb','rotate':False})


for fl in input_files:
    source_map_file = SOURCE_DATA_FOLDER+fl["map"]
    source_pdb_file = SOURCE_DATA_FOLDER+fl["pdb"]

    if fl["rotate"]:
        for k in range(N):
            euler_angles = random.random(3)*30
            target_map_file = TARGET_DATA_FOLDER+fl["map"][:-4]+'_rot{}.mrc'.format(k)
            target_pdb_file = TARGET_DATA_FOLDER+fl["pdb"][:-4]+'_rot{}.pdb'.format(k)

            EMmaps.rotate_map_and_pdb(source_map_file,source_pdb_file,euler_angles,target_map_file,target_pdb_file)
    else:
        target_map_file = TARGET_DATA_FOLDER+fl["map"][:-4]+'_rot{}.mrc'.format(0)
        target_pdb_file = TARGET_DATA_FOLDER+fl["pdb"][:-4]+'_rot{}.pdb'.format(0)
        copyfile(source_map_file,target_map_file)
        copyfile(source_pdb_file,target_pdb_file)
