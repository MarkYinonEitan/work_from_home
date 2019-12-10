#CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera-1.10"

CHIMERA_BIN="/usr/local/bin/chimera-1.13"

CHIMERA_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/NNproject/code/chimera/create_syth_dataset.py"
CHIMERA_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/NNproject/code/chimera/createDB.py"



# RESOLUTION 2.3
INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/input_files/res22/emd-2984.map"
OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/upload/emd-2984.gz.pkl"

# RESOLUTION 2.3
INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/input_files/res22/emd-2984.map"
OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/upload/emd-2984.gz.pkl"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "
#RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $INPUT_MAP_FILE $OUTPUT_MAP_FILE"

echo $RUN_COMMAND
$RUN_COMMAND
