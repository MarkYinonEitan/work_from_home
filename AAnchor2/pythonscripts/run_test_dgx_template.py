import sys
from glob import glob1
import os
import re
import shutil
import numpy as np
import traceback

from time import gmtime, strftime
from scipy import stats

import time

utils_path = utils_path = '/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor_dgx/pythoncode/utils/'
sys.path.append(utils_path)

import dbloader, utils_project
from resultsplots import DetNetResults

PDB_FILE_LINE = "HETATM AAAA  K   RES ABBBB     XXXXXXX YYYYYYY ZZZZZZZ  0.25 PPPP           K  "


def process_input(input_file,net_string, net_weights,thr_file,out_folder,debug_file):
    N= 10**4
    n_det=0

    #create folder
    out_file_name = out_folder+'/results.pdb'
    with open(out_file_name, 'w') as fout:
        fout.write("AMINO ACID CGs FOUND by AANCHOR \n" )
    #load data
    data_mtrx,filter_matrix,C,centers,labels = dbloader.load_swap_labeled_5tuple_data(input_file)

    #create candidates
    box_centers_ijk = list(dbloader.get_box_centers_for_detection(data_mtrx,filter_matrix))

    print("DEBUG 2324", filter_matrix.shape)
    print("DEBUG 2324", data_mtrx.shape)

    print("DEBUG 2324", (np.count_nonzero(filter_matrix>0)))
    print("DEBUG 2324", (np.count_nonzero(filter_matrix>0.1)))
    print("DEBUG 2324", (np.count_nonzero(filter_matrix>0.2)))



    box_centers_ijk_by_N = [box_centers_ijk[x:x+N] for x in range(0,len(box_centers_ijk),N)]

    net = utils_project.get_net_by_string(net_string)
    net.load_weights(net_weights)
    thr_dict = utils_project.read_thr_file(thr_file)

    labels_to_names_dict = dbloader.LabelbyAAType.get_labels_to_names_dict()

    with open(debug_file, "a") as myfile:
        myfile.write("007\n")
        myfile.write("Networks Loaded\n")


    #run 3  predictions (in loop)
    n_box = 0
    for centers_ijk in box_centers_ijk_by_N:
        n_box=n_box+1

        print("dets {}, box {} of {}".format(n_det,n_box,len(box_centers_ijk_by_N)))

        with open(debug_file, "a") as myfile:
            myfile.write("Start Collecting Boxes\n")
            myfile.write(time.ctime()+"\n")
        pred_boxes, box_centers_xyz_N = dbloader.get_boxes(data_mtrx,centers_ijk,C,normalization = dbloader.Mean0Sig1Normalization)
        with open(debug_file, "a") as myfile:
            myfile.write("END Collecting Boxes\n")
            myfile.write(time.ctime()+"\n")

        pred_features = np.reshape(pred_boxes,(len(pred_boxes),11,11,11,1))

        rslts = net.predict(pred_features)

        with open(debug_file, "a") as myfile:
            myfile.write("Predicitions Done\n")
            myfile.write(time.ctime()+"\n")

        det_res = DetNetResults(res_folder =out_folder, labels_names=dbloader.LabelbyAAType.get_labels_to_names_dict(), xyz=box_centers_xyz_N,results=rslts,name = 'all')

        lbs_for_detect = list(set(dbloader.LabelbyAAType.label_dict.values())-set([0]))
        res = det_res.calc_results_for_labels(lbs_for_detect)

        if len(res["dets_f"])>0:
            with open(out_file_name, 'a') as fout:
                for in_det in range(len(res["dets_f"])):
                    aa = labels_to_names_dict[res["dets_f"][in_det]]
                    min_prob = thr_dict[aa]
                    print("DEBUG 32", aa, min_prob, in_det,res["probs_f"][in_det] )
                    if res["probs_f"][in_det]>=min_prob:
                        s = str(PDB_FILE_LINE)
                        #position
                        s=s.replace('XXXXXXX', '{:3.4g}'.format(res["xyz_f"][in_det,0]).ljust(7))
                        s=s.replace('YYYYYYY', '{:3.4g}'.format(res["xyz_f"][in_det,1]).ljust(7))
                        s=s.replace('ZZZZZZZ', '{:3.4g}'.format(res["xyz_f"][in_det,2]).ljust(7))
                        s=s.replace('PPPP', '{:3.4g}'.format(res["probs_f"][in_det]).ljust(4))
                        s=s.replace('RES', aa.ljust(3))
                        s=s.replace('AAAA', str(n_det).ljust(4))
                        s=s.replace('BBBB', str(n_det).ljust(4))
                        fout.write(s)
                        fout.write("\n")
                        n_det=n_det+1

        with open(debug_file, "a") as myfile:
            myfile.write("D002 \n")

if __name__ == "__main__":

    #OUT_FOLDER="$WOKR_FOLDER//$D//"


    input_file  = "XXX_INPUT"
    net_string  = "XXX_NET_STRING"
    net_weights = "XXX_NET_WEIGHTS"
    thr_file    = "XXX_THR_FILE"
    out_folder  = "XXX_OUT_FOLDER"
    debug_file  = "XXX_DEBUG_FILE"

    process_input(input_file,net_string, net_weights, thr_file,out_folder,debug_file)
