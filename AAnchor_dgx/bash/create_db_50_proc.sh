#!/bin/bash
CHIMERA_FOLDER='../chimeracode/datasets/'
SCRIPT_NAME="db_corners_real2931_rot10.py"

cd $CHIMERA_FOLDER

for z in {0..80}
do
echo $
s1=$(($z*50))
s2=$(($s1+50))
echo $s1 $s2
chimera-1.13 --nogui  $SCRIPT_NAME  $s1  $s2&
done
