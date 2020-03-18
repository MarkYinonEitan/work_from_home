LIST_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/list_3A.txt"
VOX_FOLDER="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/vx_data/"
OUT_FOLDER="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/output/"
NET_STRING="gan_v1"
NUM_EPOCHS="30"
RESOLUTION="3.0"
VX_SIZE="1.0"


export LD_LIBRARY_PATH="/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-10.0-v7/lib64"
PYTHON_ENV="//specific//netapp5_2//iscb//wolfson//Mark//v_env//p36_tf//"
PYTHON_SCRIPT="//specific//netapp5_2//iscb//wolfson/Mark//git//work_from_home//AAcryoGAN3//code//python_scripts//run_train.py"


PYTHON_BIN="$PYTHON_ENV//bin/python3.6"


RUN_COMMAND="$PYTHON_BIN $PYTHON_SCRIPT $LIST_FILE $VOX_FOLDER $OUT_FOLDER $NET_STRING $NUM_EPOCHS $RESOLUTION $VX_SIZE"
echo $RUN_COMMAND

$RUN_COMMAND
# create input data voxalization
#create npy
#transform to map
