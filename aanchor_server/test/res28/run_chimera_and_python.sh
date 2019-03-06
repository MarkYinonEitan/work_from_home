#!/bin/bash

EM_FILE="emd-6224.map"
MAIL_ID="marik.s79@gmail.com"
FOLD_NAME="res2729"



CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="/specific/a/home/cc/cs/ppdock/webserver/AAnchor/code/pythoncode/server/create_input_file_chimera.py"


# RESOLUTION 2.8
RESOLUTION="2.8"

INPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/input_files/$FOLD_NAME/$EM_FILE"
OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/aanchor_server/upload/${EM_FILE}.gz.pkl"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $INPUT_MAP_FILE $OUTPUT_MAP_FILE"

echo $RUN_COMMAND
$RUN_COMMAND

PYTHON_BIN="python2.7"

PYTHON_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/code/pythoncode/server/run_server.py"

INPUT_GZ_FILE=$OUTPUT_MAP_FILE


RUN_COMMAND="$PYTHON_BIN  $PYTHON_SCRIPT $INPUT_GZ_FILE $RESOLUTION $MAIL_ID"


source /Users/markroza/Documents/work_from_home/data/aanchor_server/v_env/p27/bin/activate
echo $RUN_COMMAND
$RUN_COMMAND
deactivate
