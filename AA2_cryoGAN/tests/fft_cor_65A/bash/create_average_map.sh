#!/bin/bash

CHIMERA_BIN="/Applications/Chimera.app/Contents/Resources/bin/chimera"


TEMP_MAP1_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/temp/m1"
TEMP_MAP2_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/temp/m2"

OUT_MAP_FILE="/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/gan_map_average.mrc"


CHIMERA_SCRIPT="../chimera/test_fit.py"

RUN_COMMAND="$CHIMERA_BIN --nogui  $CHIMERA_SCRIPT "



#generate temp npy matrices

#average all maps in python Chimera

#save as map in Chimera

echo $RUN_COMMAND
$RUN_COMMAND
