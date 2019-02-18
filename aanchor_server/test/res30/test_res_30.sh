#!/bin/bash


CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/code/pythoncode/server/show_analyze_results.py"


OUT_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res30/results.pdb"
REF_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res30/5gaq.pdb"
RES_FILE="/Users/markroza/Documents/GitHub/work_from_home/aanchor_server/test/res30/results.txt"



RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test $OUT_FILE $REF_FILE $RES_FILE"

echo $RUN_COMMAND
$RUN_COMMAND


RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE GLY 0.9"
$RUN_COMMAND
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE PRO 0.75"
$RUN_COMMAND
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE ARG 0.6"
$RUN_COMMAND
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE LEU 0.75"
$RUN_COMMAND
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE LYS 0.7"
$RUN_COMMAND
RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT test_one $OUT_FILE $REF_FILE TYR 0.00"
$RUN_COMMAND
