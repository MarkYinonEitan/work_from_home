#!/bin/bash


CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="code/pythoncode/server/create_input_file_chimera.py"

# RESOLUTION 2.3
RESOLUTION="2.3"
INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/input_files/res22/emd-2984.map"
OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/upload/emd-2984.gz.pkl"

# RESOLUTION 2.8
#RESOLUTION="2.8"

#INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/input_files/res2931/emd-8015.mrc"
#OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/upload/emd-8015.gz.pkl"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $INPUT_MAP_FILE $OUTPUT_MAP_FILE"

echo $RUN_COMMAND
$RUN_COMMAND

PYTHON_BIN="python2.7"

PYTHON_SCRIPT="code/pythoncode/server/run_server.py"

INPUT_GZ_FILE=$OUTPUT_MAP_FILE


RUN_COMMAND="$PYTHON_BIN  $PYTHON_SCRIPT $INPUT_GZ_FILE $RESOLUTION marik.s79@gmail.com"


source /Users/markroza/Documents/work_from_home/data/aanchor_server/v_env/p27/bin/activate
echo $RUN_COMMAND
$RUN_COMMAND
deactivate
