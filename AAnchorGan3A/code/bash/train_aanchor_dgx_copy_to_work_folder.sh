
SOURCE_FOLDER="$(pwd)/"

LIST_FILE="$SOURCE_FOLDER/list_3A_after_rotation.txt"
DB_FOLDER="$SOURCE_FOLDER/DB2931class_rot10/"
D=$(date "+DATE_%m_%d_%y_TIME_%H_%M_%S")
WORK_FOLDER="$SOURCE_FOLDER/train_res/"
NET_STRING="V5_DROP_REG_2"
WEIGHTS_FILE="NOTHING"
NUM_EPOCHS="300"

OUT_FOLDER="$WORK_FOLDER//AANCHOR_$D_$NET_STRING//"
PYTHON_SCRIPT_TEMP="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchorGan3A/code/python_scripts/run_train_on_dgx_template.py"

PYTHON_SCRIPT="$OUT_FOLDER//run_train_on_dgx.py"
#
#
# CREATE INPUT FOLDER
rm -rf $OUT_FOLDER
mkdir $OUT_FOLDER
cp $PYTHON_SCRIPT_TEMP $PYTHON_SCRIPT

sed -i "s|XXX_INPUT_LIST|$LIST_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_INPUT_DB_FOLDER|$DB_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_OUT_FOLDER|$OUT_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_NET_STRING|$NET_STRING|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_NET_WEIGHTS|$WEIGHTS_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|XXX_N_EPOCHS|$NUM_EPOCHS|g" "$PYTHON_SCRIPT"

RUN_COMMAND="srun -G 1 --pty easy_ngc --modules keras --train "$PYTHON_SCRIPT"  tensorflow "
ssh op-controller.cs.tau.ac.il $RUN_COMMAND
ssh op-controller.cs.tau.ac.il pkill -u markroza
