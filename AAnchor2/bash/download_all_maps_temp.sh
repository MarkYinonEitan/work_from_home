#!/bin/bash

FTP_SERVER=' ftp.ebi.ac.uk'

PYTHON_FOLDER='../pythoncode/scripts/'
SCRIPT_NAME="create_db_res3_apix_var.py"

cd $PYTHON_FOLDER

python2.7 $SCRIPT_NAME "0" "400" &
python2.7 $SCRIPT_NAME "400" "800" &
python2.7 $SCRIPT_NAME "800" "1200" &
python2.7 $SCRIPT_NAME "1200" "1600" &
python2.7 $SCRIPT_NAME "1600" "2000" &
python2.7 $SCRIPT_NAME "2000" "2400" &
python2.7 $SCRIPT_NAME "2400" "2800" &
python2.7 $SCRIPT_NAME "2800" "3200" &
python2.7 $SCRIPT_NAME "3200" "3600" &
python2.7 $SCRIPT_NAME "3600" "4000" &

