#!/bin/bash


CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="code/pythoncode/server/create_input_file_chimera.py"


RES_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/input_files/res22/emd-2984.map"
REF_PDB_FILE=""
INPUT_EM_MAP

OUTPUT_MAP_FILE="/Users/markroza/Documents/work_from_home/aanchor_server/upload/emd-2984.gz.pkl"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT $INPUT_MAP_FILE $OUTPUT_MAP_FILE"

echo $RUN_COMMAND
$RUN_COMMAND
