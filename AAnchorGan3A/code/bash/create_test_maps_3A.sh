LIST_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/test1/list_3A_test.txt"
PDBS_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/res2931/"
VOX_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/test1/vx/"
OUT_NPY="//specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/test1/out_npy/"
OUT_MRC="//specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/test1/out_mrc/"
WEIGHTS_FILE="/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/saved_results/800000.ckpt"
NET_STRING="gan_mean_sigma"
RESOLUTION="3.0"
VX_SIZE="1.0"

#CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
CHIMERA_BIN="/usr/local/bin/chimera-1.13"

CHIMERA_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/chimera_scripts/vox_from_list_3A.py"


VX_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $LIST_FILE $PDBS_FOLDER $VOX_FOLDER"

echo $VX_COMMAND
$VX_COMMAND




export LD_LIBRARY_PATH="/usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-10.0-v7/lib64"
PYTHON_ENV="//specific//netapp5_2//iscb//wolfson//Mark//v_env//p36_tf//"
PYTHON_BIN="$PYTHON_ENV//bin/python3.6"
PYTHON_SCRIPT="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/python_scripts/create_npy_map_from_list.py"

GAN_COMMAND="$PYTHON_BIN $PYTHON_SCRIPT $LIST_FILE $VOX_FOLDER $OUT_NPY $NET_STRING $WEIGHTS_FILE $RESOLUTION $VX_SIZE"

echo $GAN_COMMAND
$GAN_COMMAND

NPY_TO_MAP_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/chimera_scripts/matrix_to_map.py"

NPY_TO_MAP_COMMAND="$CHIMERA_BIN --nogui  $NPY_TO_MAP_SCRIPT $LIST_FILE $OUT_NPY  $OUT_MRC $RESOLUTION $VX_SIZE"
echo $NPY_TO_MAP_COMMAND
$NPY_TO_MAP_COMMAND
