#!/bin/bash


#INPUTS : PDB_FILE, GAN_FILE, DISCCR_FILE, OUT_MAP FILE

#PARAMETERS:
CHIMERA_BIN=""
PYTHON_SCRIPT=""

source /Users/markroza/Documents/work_from_home/NNcourse_project/v_env_p37/bin/activate
python3.7 ../python/create_one_6n18_maps.py 

# create input data voxalization
#create npy
#transform to map
