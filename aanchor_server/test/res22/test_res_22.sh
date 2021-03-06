#!/bin/bash


CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/code/pythoncode/server/show_analyze_results.py"


OUT_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res22/results.pdb"
REF_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res22/pdb5a1a.ent.txt"
RES_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res22/results.txt"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test $OUT_FILE $REF_FILE $RES_FILE"

echo $RUN_COMMAND
$RUN_COMMAND
