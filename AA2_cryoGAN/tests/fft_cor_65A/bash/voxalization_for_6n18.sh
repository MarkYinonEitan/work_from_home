#!/bin/bash

CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"

CHIMERA_SCRIPT="../chimera/vox6nt8.py"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "

echo $RUN_COMMAND
$RUN_COMMAND
