
SOURCE_FOLDER="$(pwd)/"

LIST_FILE="$SOURCE_FOLDER/list_3A_after_rotation.txt"
WEIGHTS_FILE="$SOURCE_FOLDER/saved_net_results/gan_first_run/22300.ckpt"
VOX_FOLDER="$SOURCE_FOLDER/DB2931class_rot10/"
D=$(date "+DATE_%m_%d_%y_TIME_%H_%M_%S")
WORK_FOLDER="$SOURCE_FOLDER/train_res/"
NET_STRING="gan_mean_sigma"
OUT_FOLDER="$WORK_FOLDER//TEST_GAN_${D}_${NET_STRING}//"
RESOLUTION_INP="3.0"
VX_SIZE_INP="1.0"
PYTHON_SCRIPT_TEMP="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchorGan3A/code/python_scripts/test_gan_dgx_template.py"

PYTHON_SCRIPT="$OUT_FOLDER//test_gan_dgx.py"
#
#
# CREATE INPUT FOLDER
rm -rf $OUT_FOLDER
mkdir $OUT_FOLDER
cp $PYTHON_SCRIPT_TEMP $PYTHON_SCRIPT

sed -i "s|LIST_FILE|$LIST_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|VOX_FOLDER|$VOX_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|OUT_FOLDER|$OUT_FOLDER|g" "$PYTHON_SCRIPT"
sed -i "s|NET_STRING|$NET_STRING|g" "$PYTHON_SCRIPT"
sed -i "s|WEIGHTS_FILE|$WEIGHTS_FILE|g" "$PYTHON_SCRIPT"
sed -i "s|RESOLUTION_INP|$RESOLUTION_INP|g" "$PYTHON_SCRIPT"
sed -i "s|VX_SIZE_INP|$VX_SIZE_INP|g" "$PYTHON_SCRIPT"


PYTHON_BIN="/specific/netapp5_2/iscb/wolfson/Mark/v_env/p36_tf/bin/python"

export LD_LIBRARY_PATH="/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-9.0-v7/lib64"
$PYTHON_BIN $PYTHON_SCRIPT
