import sys
from glob import glob1
import os
import re
import shutil
import numpy as np
from mail import zip_and_send_mail

from time import gmtime, strftime
from scipy import stats

import time

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
nets_path = dir_path + '/../nets/'

sys.path.append(utils_path)
sys.path.append(nets_path)

import dbloader, all_nets
from resultsplots import DetNetResults

try:
  import chimera
except ImportError:
  print "RUN Without Chimera"


DEBUG_FILE = "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/temp/temp_python.txt"

PDB_FILE_LINE = "HETATM AAAA  K   RES ABBBB     XXXXXXX YYYYYYY ZZZZZZZ  0.25 14.49           K  "


#WORK_DIR = "/specific/netapp5_2/iscb/wolfson/Mark/Projects/NNcryoEM/data/temp"
#UPLOAD_DIR = "/specific/netapp5_2/iscb/wolfson/Mark/Projects/NNcryoEM/data/temp"

WORK_DIR = "/specific/disk1/webservertmp/AAnchor/upload/"
UPLOAD_DIR = "/specific/netapp5_2/iscb/wolfson/Mark/Projects/NNcryoEM/data/temp"

NETS_FOLDER = "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/bin/nets_data/"


def get_thr_data(res):

    thr_data=[]

    if res == "2.3":
        thr_data.append({"AA":"ARG","combination":"maj3","thr":0.3})
        thr_data.append({"AA":"LEU","combination":"mean3","thr":0.3})
        thr_data.append({"AA":"PRO","combination":"NES","thr":0.65})
        thr_data.append({"AA":"VAL","combination":"mean3","thr":0.3})

    if res == "2.8":
        thr_data.append({"AA":"ASN","combination":"NE","thr":0.35})
        thr_data.append({"AA":"ARG","combination":"NE","thr":0.65})
        thr_data.append({"AA":"LEU","combination":"NE","thr":0.7})
        thr_data.append({"AA":"LYS","combination":"mean2","thr":0.45})
        thr_data.append({"AA":"PRO","combination":"maj3","thr":0.5})
        thr_data.append({"AA":"TYR","combination":"maj3","thr":0.35})
        thr_data.append({"AA":"VAL","combination":"NE","thr":0.35})

    if res == "3.1":
        thr_data.append({"AA":"ARG","combination":"NES","thr":0.6})
        thr_data.append({"AA":"GLY","combination":"NE","thr":0.5})
        thr_data.append({"AA":"LEU","combination":"NE","thr":0.7})
        thr_data.append({"AA":"LYS","combination":"NE","thr":0.65})
        thr_data.append({"AA":"PRO","combination":"mean2","thr":0.65})
        thr_data.append({"AA":"TYR","combination":"maj3","thr":0.3})

    return thr_data


def create_three_files(folder_name):

    lines=['111','22','3']

    file1 = "f1.txt"
    file2 = "f2.txt"
    file3 = "f3.txt"


    f1 = open(folder_name+'/'+file1, "w")
    f1.write(lines[0])
    f1.close()

    f2 = open(folder_name+'/'+file2, "w")
    f2.write(lines[1])
    f2.close()

    f3 = open(folder_name+'/'+file3, "w")
    f3.write(lines[2])
    f3.close()

    f_names = []
    f_names.append(folder_name+file1)
    f_names.append(folder_name+file2)
    f_names.append(folder_name+file3)

    print (folder_name+file1)

    return f_names





def create_results_folder(file_name):

    file_name_corrected = re.sub("[^a-zA-Z0-9]+", "", file_name)

    full_fold_name = WORK_DIR + '/'+file_name_corrected+strftime("Y%YM%mD%dH%HMMS%S",  gmtime());

    try:
        os.makedirs(full_fold_name)

    except OSError as e:
        if not os.path.isdir(full_fold_name):
            with open(DEBUG_FILE, "a") as myfile:
                myfile.write("0015\n")
            with open(DEBUG_FILE, "a") as myfile:
                myfile.write("I/O error({0}): {1}".format(e.errno, e.strerror))
            raise
    return full_fold_name

def send_mail(files, email):
    return

def calc_mean2(res1,res2):
    res_data = (res1 + res2)/2.0
    return res_data

def calc_mean3(res1,res2,res3):
    res_data = (res1 + res2+res3)/2.0
    return res_data

def calc_maj3(res1,res2,res3):

    dets1 = np.argmax(res1,axis=1).reshape(-1,1)
    dets2 = np.argmax(res2,axis=1).reshape(-1,1)
    dets3 = np.argmax(res3,axis=1).reshape(-1,1)
    dets_array = np.hstack((dets1,dets2,dets3))
    majority = stats.mode(dets_array,axis=1)
    majority.mode[majority.count==0] = 0

    res_new = np.zeros(res1.shape)
    for k in range(res_new.shape[0]):
        if majority.mode[k] == 0:
            continue

        n=0
        if dets1[k] == majority.mode[k]:
            res_new[k,:] = res_new[k,:]+dets1[k,:]
            n=n+1
        if dets2[k] == majority.mode[k]:
            res_new[k,:] = res_new[k,:]+dets2[k,:]
            n=n+1
        if dets3[k] == majority.mode[k]:
            res_new[k,:] = res_new[k,:]+dets2[k,:]
            n=n+1

        res_new[k,:] = res_new[k,:]/n

    res_data = (res1 + res2+res3)/2.0
    return res_new


def process_input(full_path_file_name,res,email_ad):
    N= 10**3
    res_folder={"2.3":"res0023","2.8":"res2528","3.1":"res2931",}
    n_det=0

    #create folder
    file_name = os.path.basename(full_path_file_name)
    out_folder_name = create_results_folder(file_name)
    out_file_name = out_folder_name+'/results.pdb'
    with open(out_file_name, 'w') as fout:
        fout.write("AMINO ACID CGs FOUND by AANCHOR \n" )
    #load data
    data_mtrx,filter_matrix,C,centers,labels = dbloader.load_swap_labeled_5tuple_data(full_path_file_name)

    #create candidates
    box_centers_ijk = dbloader.get_box_centers_for_detection(data_mtrx,filter_matrix)

    box_centers_ijk_by_N = [box_centers_ijk[x:x+N] for x in range(0,len(box_centers_ijk),N)]

    nets={}
    nets["NE"] = all_nets.V5_no_reg().get_compiled_net()
    nets["NS"] = all_nets.V5_no_reg().get_compiled_net()
    nets["NES"] = all_nets.V5_no_reg().get_compiled_net()
    weights_file_ne = NETS_FOLDER+res_folder[res]+'/NE/'+'weights_updated.h5'

    nets["NE"].load_weights(weights_file_ne)
    weights_file_ns = NETS_FOLDER+res_folder[res]+'/NS/'+'weights_updated.h5'
    nets["NS"].load_weights(weights_file_ns)
    weights_file_nes = NETS_FOLDER+res_folder[res]+'/NES/'+'weights_updated.h5'
    nets["NES"].load_weights(weights_file_nes)


    rslts={"NE":[],"NS":[],"NES":[],'mean2':[],"mean3":[],"maj3":[],}

    with open(DEBUG_FILE, "a") as myfile:
        myfile.write("007\n")
        myfile.write("Networks Loaded\n")

    all_thr_data = get_thr_data(res)

    #run 3  predictions (in loop)
    for centers_ijk in box_centers_ijk_by_N:

        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("Start Collecting Boxes\n")
            myfile.write(time.ctime()+"\n")

        pred_boxes, box_centers_xyz_N = dbloader.get_boxes(data_mtrx,centers_ijk,C,normalization = dbloader.Mean0Sig1Normalization)
        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("END Collecting Boxes\n")
            myfile.write(time.ctime()+"\n")

        pred_features = np.reshape(pred_boxes,(len(pred_boxes),11,11,11,1))

        rslts["NE"] = nets["NE"].predict(pred_features)
        rslts["NS"] = nets["NS"].predict(pred_features)
        rslts["NES"] = nets["NES"].predict(pred_features)

        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("Predicitions Done\n")
            myfile.write(time.ctime()+"\n")

        rslts["mean2"]  = calc_mean2(rslts["NE"],rslts["NS"])
        rslts["mean3"]  = calc_mean3(rslts["NE"],rslts["NS"],rslts["NES"])
        rslts["maj3"]  = calc_maj3(rslts["NE"],rslts["NS"],rslts["NES"])


        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("Cominations Calculated Done\n")
            myfile.write(time.ctime()+"\n")



        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("D001 \n")

        for thr in all_thr_data:
            AA_NAME = thr["AA"]
            comb_name = thr["combination"]
            thr = thr["thr"]

            det_res = DetNetResults(res_folder =out_folder_name, labels_names=dbloader.LabelbyAAType.get_labels_to_names_dict(), xyz=box_centers_xyz_N,results=rslts[comb_name],name = AA_NAME)

            try:
                res = det_res.calc_results_for_labels(dbloader.LabelbyAAType.label_dict[AA_NAME])

                with open(DEBUG_FILE, "a") as myfile:
                    myfile.write(AA_NAME+str(len(res["xyz_f"]))+" \n")

                if len(res["xyz_f"])>0:
                    with open(out_file_name, 'a') as fout:
                        for in_det in range(len(res["xyz_f"])):
                            s = str(PDB_FILE_LINE)
                            #position
                            s=s.replace('XXXXXXX', '{:3.4g}'.format(res["xyz_f"][in_det,0]).ljust(7))
                            s=s.replace('YYYYYYY', '{:3.4g}'.format(res["xyz_f"][in_det,1]).ljust(7))
                            s=s.replace('ZZZZZZZ', '{:3.4g}'.format(res["xyz_f"][in_det,2]).ljust(7))
                            s=s.replace('RES', AA_NAME.ljust(3))
                            s=s.replace('AAAA', str(n_det).ljust(4))
                            s=s.replace('BBBB', str(n_det).ljust(4))
                            fout.write(s)
                            fout.write("\n")
                            n_det=n_det+1




            except:
                with open(DEBUG_FILE, "a") as myfile:
                    myfile.write(AA_NAME+" FAILED \n")

        with open(DEBUG_FILE, "a") as myfile:
            myfile.write("D002 \n")


    #calc 3 combinations

    #extract anchors

    #create output files




    #create three files

    create_three_files(out_folder_name)

    zip_and_send_mail(email_ad, out_folder_name)



if __name__ == "__main__":

    with open(DEBUG_FILE, "w") as myfile:
        myfile.write("START\n")



    input_file_full_path = sys.argv[1]
    resolution =  sys.argv[2]
    email_ad =  sys.argv[3]
    #output_folder =   sys.argv[4]

    process_input(input_file_full_path,resolution, email_ad)

    with open(DEBUG_FILE, "a") as myfile:
        myfile.write("Continue\n")

    with open(DEBUG_FILE, "a") as myfile:
        myfile.write("End\n")


    #data_lines = read_input_file(emmap_file_full_path)
    #f_names = create_three_files(data_lines,output_folder)
    #send_mail(f_names, email)
