
INP_EM_MAP="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/emd-7526.mrc"
INP_REFERENCE_PDB="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/6cmx.pdb"
RESOLUTION_INP="3.0"
VX_SIZE_INP="1.0"
NET_STRING="V5_no_reg"
NET_WEIGHTS_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/nets_data/v4_db2931_real_rot10/weights_updated_19.h5"
WOKR_FOLDER="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/temp/"
THR_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/thr_3A_20_1_2020.txt"


CHIMERA_CREATE_INP_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimerascripts/create_input_file_chimera.py"


#FOLDERS AND FILES
D=$(date "+DATE_%m_%d_%y_TIME_%H_%M_%S")
OUT_FOLDER="$WOKR_FOLDER//$D//"
DB_FILE="$OUT_FOLDER//input_db_file.pkl.gz"
PYTHON_DEBUG_FILE="$OUT_FOLDER//python_debug.txt"

OUT_TXT_FILE="$OUT_FOLDER//screen_out.txt"

# CREATE INPUT FOLDER
rm -rf $OUT_FOLDER
mkdir $OUT_FOLDER
cp $INP_EM_MAP $OUT_FOLDER
cp $INP_REFERENCE_PDB $OUT_FOLDER

echo "FOLDERS CREATED"

#CREATE INPUT FILE
#chimera-1.13 --nogui  $CHIMERA_CREATE_INP_SCRIPT  $INP_EM_MAP  $DB_FILE >> $OUT_TXT_FILE
echo "INPUT FILE CREATED"


DB_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/temp/DATE_01_14_20_TIME_16_25_34/input_db_file.pkl.gz"

export LD_LIBRARY_PATH="/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-9.0-v7/lib64"
PYTHON_BIN="/specific/netapp5_2/iscb/wolfson/Mark/v_env/p36_tf/bin/python"
PYTHON_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/pythonscripts/run_net_on_one_map.py"

echo $LD_LIBRARY_PATH
$PYTHON_BIN $PYTHON_SCRIPT $DB_FILE $NET_STRING $NET_WEIGHTS_FILE $THR_FILE $OUT_FOLDER $PYTHON_DEBUG_FILE

#PYTHON_SCRIPT="\/\/specific\/\/netapp5_2\/\/iscb\/\/wolfson\/Mark\/\/git\/\/work_from_home\/\/AAcryoGAN3\/\/code\/\/python_scripts\/\/run_train.py"
# PYTHON_SCRIPT_TEMP="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/python_scripts/run_train_dgx.py"


#PYTHON_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/python_scripts/dgx_train_scripts/train_$D.py"

# PYTHON_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/pythonscripts/run_training.py"


#cp $PYTHON_SCRIPT_TEMP $PYTHON_SCRIPT

#sed -i "s|LIST_FILE|$LIST_FILE|g" "$PYTHON_SCRIPT"
#sed -i "s|VOX_FOLDER|$VOX_FOLDER|g" "$PYTHON_SCRIPT"
#sed -i "s|OUT_FOLDER|$OUT_FOLDER|g" "$PYTHON_SCRIPT"
#sed -i "s|NET_STRING|$NET_STRING|g" "$PYTHON_SCRIPT"
#sed -i "s|NUM_EPOCHS|$NUM_EPOCHS|g" "$PYTHON_SCRIPT"
#sed -i "s|RESOLUTION_INP|$RESOLUTION_INP|g" "$PYTHON_SCRIPT"
#sed -i "s|VX_SIZE_INP|$VX_SIZE_INP|g" "$PYTHON_SCRIPT"


# RUN_COMMAND="srun -G 1 --pty easy_ngc --modules keras --train "$PYTHON_SCRIPT"  tensorflow "
# echo $RUN_COMMAND
#$RUN_COMMAND
