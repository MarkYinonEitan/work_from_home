#!/bin/bash
#PARAMETERS:
CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
PYTHON_BIN="python3.7"

ENV_BIN="/Users/markroza/Documents/work_from_home/NNcourse_project/v_env_p37/bin/activate"

VOXALIZATION_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera_scripts/create_input_matrices.py"
DISCRIMINATOR_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/python_scripts/run_discriminator.py"
MATRIX_TO_MAP_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera_scripts/matrix_to_map.py"


#INPUTS : PDB_FILE, GAN_FILE, DISCCR_FILE, OUT_MAP FILE
RESOLUTION="6.0"
PDB_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8.pdb"
MAP_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8_molmap.mrc"
DISC_WEIGHTS_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/nets/7100.ckpt"
VOX_SIZE="2.0"
WORK_FOLDER="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/temp/"
PDB_ID="6nt8"
RESULT_NPY_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8_molmap_disc.npy"
RESULT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8_molmap_disc.mrc"
DISCR_WEGHTS_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/nets/7100.ckpt"
DISC_NET="disc_v1"


# RUN VOXALIZATION
RUN_COMMAND="$CHIMERA_BIN --nogui $VOXALIZATION_SCRIPT $PDB_ID $PDB_FILE $MAP_FILE $WORK_FOLDER $VOX_SIZE $RESOLUTION"
echo $RUN_COMMAND
$RUN_COMMAND


#RUN DISCRIMINATOR
RUN_COMMAND="$PYTHON_BIN $DISCRIMINATOR_SCRIPT $WORK_FOLDER $PDB_ID $RESULT_NPY_FILE $DISC_NET $DISCR_WEGHTS_FILE"
echo $RUN_COMMAND
source $ENV_BIN
$RUN_COMMAND
deactivate

# RUN VOXALIZATION
RUN_COMMAND="$CHIMERA_BIN --nogui $MATRIX_TO_MAP_SCRIPT $RESULT_NPY_FILE $RESULT_MAP_FILE"
echo $RUN_COMMAND
$RUN_COMMAND
