CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
#CHIMERA_BIN="/usr/local/bin/chimera-1.13"
WORK_FOLD="/Users/markroza/Documents/GitHub/work_from_home/NNproject/"




CHM_MTRX2MAP="$WORK_FOLD/code/chimera/matrix_to_map.py"

COMMAND="mtrx2map"
VX_SIZE="2.0"
INP_FILE="/Users/markroza/Documents/work_from_home/NNcourse_project/data/results/final_res6/test_6nt8.npy"
MAP_NAME="6nt8_s"
OUT_FLD="/Users/markroza/Documents/temp/"
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHM_MTRX2MAP $COMMAND $VX_SIZE $INP_FILE $MAP_NAME $OUT_FLD"

echo $RUN_COMMAND
$RUN_COMMAND
