CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"
#CHIMERA_BIN="/usr/local/bin/chimera-1.13"

#CHIMERA_SCRIPT="/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/NNproject/code/chimera/createDB.py"
CHIMERA_SCRIPT="/Users/markroza/Documents/GitHub/work_from_home/AAcryoGAN3Angstrem/code/chimera/createDB.py"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "

echo $RUN_COMMAND
$RUN_COMMAND