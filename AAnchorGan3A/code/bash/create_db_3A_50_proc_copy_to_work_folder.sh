#!/bin/bash

echo "PUT THIS FILE IN WORK FOLDER"

INP_LIST_FILE="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchorGan3A/runs_sh_and_list_files/real_maps_3A_first_run/list_3A_after_rotation.txt"
ERR_LIST_FILE="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchorGan3A/runs_sh_and_list_files/real_maps_3A_first_run/list_3A_DB_errors.txt"

INPUT_DATA_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/cryoEM/raw_data/res2931mrc_after_rotation/"
OUT_DB_FOLDER="//specific/netapp5_2/iscb/wolfson/Mark/data/AAnchorGan3A/runs_sh_and_list_files/real_maps_3A_first_run/DB2931class_rot10/"

CHIMERA_CREATE_SCRIPT="//specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchorGan3A/code/chimera_scripts/create_db_corners_n_m.py"

rm -rf $ERR_LIST_FILE || true
for z in {0..80}
do
echo $
s1=$(($z*20)) # s1=$(($z*50))
s2=$(($s1+20)) # s2=$(($s1+50))
echo $s1 $s2
chimera-1.13 --nogui  $CHIMERA_CREATE_SCRIPT $INP_LIST_FILE $INPUT_DATA_FOLDER $OUT_DB_FOLDER $ERR_LIST_FILE $s1 $s2&
done
