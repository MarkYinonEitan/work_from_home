import chimera
from chimera import runCommand
import chimera
import VolumeViewer
import glob
import numpy as np
import os, sys
import time
import shutil


#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path +'/../python/'
chimera_path = dir_path +'/../chimera/'

sys.path.append(python_path)
sys.path.append(chimera_path)

import createDB


if __name__ == "chimeraOpenSandbox":
    k=0
    while os.path.basename(sys.argv[k])[0:5]!= __file__[0:5]:
        k=k+1
        print sys.argv[k], '###', __file__
    list_file = sys.argv[k+1]
    input_folder = sys.argv[k+2]
    output_folder = sys.argv[k+3]
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)

    os.mkdir(output_folder)
    createDB.create_database(input_folder, output_folder, list_file)
    runCommand('stop')
