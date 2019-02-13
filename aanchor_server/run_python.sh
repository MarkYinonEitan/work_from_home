#!/bin/bash


PYTHON_BIN="python2.7"

PYTHON_SCRIPT="code/pythoncode/server/run_server.py"

INPUT_GZ_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/upload/emd-2984.gz.pkl"
RESOLUTION="2.3"



RUN_COMMAND="$PYTHON_BIN  $PYTHON_SCRIPT $INPUT_GZ_FILE $RESOLUTION marik.s79@gmail.com"


source v_env/p27/bin/activate
echo $RUN_COMMAND
$RUN_COMMAND
deactivate
