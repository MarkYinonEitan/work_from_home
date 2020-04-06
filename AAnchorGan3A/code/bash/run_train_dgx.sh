


LIST_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/list_3A.txt"
VOX_FOLDER="\/specific\/netapp5_2\/iscb\/wolfson\/Mark\/data\/AAcryoGaN3\/vx_data\/"
D=$(date "+DATE_%m_%d_%y_TIME_%H_%M_%S")
OUT_FOLDER="$WOKR_FOLDER//$D//"
OUT_FOLDER="\/specific\/netapp5_2\/iscb\/wolfson\/Mark\/data\/AAcryoGaN3\/output\/"
NET_STRING="gan_mean_sigma"
NUM_EPOCHS="30"
RESOLUTION_INP="3.0"
VX_SIZE_INP="1.0"


#PYTHON_SCRIPT="\/\/specific\/\/netapp5_2\/\/iscb\/\/wolfson\/Mark\/\/git\/\/work_from_home\/\/AAcryoGAN3\/\/code\/\/python_scripts\/\/run_train.py"
D=$(date "+DATE_%m_%d_%y_TIME_%H_%M_%S")
PYTHON_SCRIPT_TEMP="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/python_scripts/run_train_dgx.py"
PYTHON_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/python_scripts/dgx_train_scripts/train_$D.py"


cp $PYTHON_SCRIPT_TEMP $PYTHON_SCRIPT

sed -i "s|LIST_FILE|$LIST_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|VOX_FOLDER|$VOX_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|OUT_FOLDER|$OUT_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|NET_STRING|$NET_STRING|g" "$PYTHON_SCRIPT"
sed -i "s|NUM_EPOCHS|$NUM_EPOCHS|g" "$PYTHON_SCRIPT"
sed -i "s|RESOLUTION_INP|$RESOLUTION_INP|g" "$PYTHON_SCRIPT"
sed -i "s|VX_SIZE_INP|$VX_SIZE_INP|g" "$PYTHON_SCRIPT"


RUN_COMMAND="srun -G 1 --pty easy_ngc --modules keras --train "$PYTHON_SCRIPT"  tensorflow "
echo $RUN_COMMAND
$RUN_COMMAND
