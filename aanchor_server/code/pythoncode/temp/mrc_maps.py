import os
import sys
import time
import numpy as np
import random


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
ch_path = dir_path + '/../../chimeracode/'
sys.path.append(utils_path)
sys.path.append(ch_path)

import resultsplots
reload(resultsplots)
from resultsplots import DetNetResults,DetRes4Nets
NETS_FOLDER = dir_path +'/../../data/nets_data/'
BASE_FOLDER = NETS_FOLDER+'/det_analysis_2931/'
dtrs1 = DetNetResults(res_folder = BASE_FOLDER+'/mean_of_three/' ,name = '  ')

dtrs1.load_data()

for x in range(1,21):
    dtrs1.plot_results_per_label([x],N=10000)
    print dtrs1.labels_names[x]
