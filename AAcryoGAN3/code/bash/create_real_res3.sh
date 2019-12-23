#CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
CHIMERA_BIN="/usr/local/bin/chimera-1.13"

CHIMERA_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAcryoGAN3/code/chimera/createDB.py"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "

echo $RUN_COMMAND
$RUN_COMMAND
