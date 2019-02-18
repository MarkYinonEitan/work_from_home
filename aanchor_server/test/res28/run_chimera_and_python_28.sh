#!/bin/bash


CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
test/res28/test_res_28.sh

CHIMERA_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/code/pythoncode/server/create_input_file_chimera.py"


# RESOLUTION 2.8
RESOLUTION="2.8"

INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/input_files/res2729/emd-6224.map"
OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/upload/emd-6224.gz.pkl"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $INPUT_MAP_FILE $OUTPUT_MAP_FILE"

echo $RUN_COMMAND
$RUN_COMMAND

PYTHON_BIN="python2.7"

PYTHON_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/code/pythoncode/server/run_server.py"

INPUT_GZ_FILE=$OUTPUT_MAP_FILE


RUN_COMMAND="$PYTHON_BIN  $PYTHON_SCRIPT $INPUT_GZ_FILE $RESOLUTION marik.s79@gmail.com"


source /Users/markroza/Documents/work_from_home/data/aanchor_server/v_env/p27/bin/activate
echo $RUN_COMMAND
$RUN_COMMAND
deactivate
