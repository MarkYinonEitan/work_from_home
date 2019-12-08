#!/bin/bash

CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="../chimera/test_fit.py"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "

echo $RUN_COMMAND
$RUN_COMMAND
