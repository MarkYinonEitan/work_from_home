INP_EM_MAP="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/emd-7526.mrc"
INP_REFERENCE_PDB="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931/6cmx.pdb"
RESOLUTION_INP="3.0"
VX_SIZE_INP="1.0"
NET_STRING="V5_no_reg"
NET_WEIGHTS_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/nets_data/v4_db2931_real_rot10/weights_updated_19.h5"
WOKR_FOLDER="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/temp/"
THR_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/thr_3A_20_1_2020.txt"
SIGMA_THR="0.001"


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
chimera-1.13 --nogui  $CHIMERA_CREATE_INP_SCRIPT  $INP_EM_MAP $DB_FILE $SIGMA_THR>> $OUT_TXT_FILE
echo "INPUT FILE CREATED"

PYTHON_SCRIPT_TEMP="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/pythonscripts//run_test_dgx_template.py"
PYTHON_SCRIPT="$OUT_FOLDER//run_test_dgx.py"
RES_PDB_FILE="$OUT_FOLDER//results.pdb"
RES_TXT_FILE="$OUT_FOLDER//results.txt"


cp $PYTHON_SCRIPT_TEMP $PYTHON_SCRIPT

sed -i "s|XXX_INPUT|$DB_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_NET_STRING|$NET_STRING|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_NET_WEIGHTS|$NET_WEIGHTS_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_THR_FILE|$THR_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_OUT_FOLDER|$OUT_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_DEBUG_FILE|$PYTHON_DEBUG_FILE|g" "$PYTHON_SCRIPT"


RUN_COMMAND="srun -G 1 --pty easy_ngc --modules keras --train "$PYTHON_SCRIPT"  tensorflow "
echo $RUN_COMMAND
ssh op-controller.cs.tau.ac.il $RUN_COMMAND
ssh op-controller.cs.tau.ac.il pkill -u markroza

#ANALYSE FILE
CHIMERA_ANALYZE_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/chimerascripts/show_analyze_results.py"
CHIMERA_COMMAND="test_graphs"
chimera-1.13 --nogui  $CHIMERA_ANALYZE_SCRIPT $CHIMERA_COMMAND $RES_PDB_FILE $INP_REFERENCE_PDB $RES_TXT_FILE > $OUT_TXT_FILE
