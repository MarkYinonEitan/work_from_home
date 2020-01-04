import os
import sys
import time
from glob import glob1
import numpy as np
import subprocess
from scipy import stats

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
nets_path = dir_path + '/../nets/'
sys.path.append(utils_path)
sys.path.append(nets_path)

import all_nets
import dbloader
from dbloader import  Mean0Sig1Normalization,load_swap_labeled_5tuple_data,get_box_centers_for_detection,get_boxes,LabelbyAAType
import resultsplots
reload(resultsplots)
from resultsplots import DetNetResults,DetRes4Nets
CHIMERA_SCRIPT = dir_path +'/../../chimeracode/create_det_set_labeled.py'

class MeanProbConsensus(object):
    @staticmethod
    def calc_consensus(res_data_per_net):
        N = len(res_data_per_net)
        res_data = res_data_per_net[0]
        for k in range(1,N):
            res_data = res_data+res_data_per_net[k]
        res_data = res_data/N
        return res_data

class MajorityConsensus(object):
    @staticmethod
    def calc_consensus(res_data_per_net):

        dets_per_net = [np.argmax(res,axis=1) for res in res_data_per_net]
        dets_array = np.hstack(zip(dets_per_net))
        print "DEBUB 1", dets_array.shape
        majority = stats.mode(dets_array,axis=1)
        majority.mode[majority.count==0] = 0

        res_new = np.zeros(res_data_per_net.shape)
        for k in range(res_new.shape[0]):
            if majority.mode[k] == 0:
                continue
            n=0
            for in_res in range(dets_per_net):
                if dets_per_net[in_res][k] == majority.mode[k]:
                    res_new[k,:] = res_new[k,:]+ res_data_per_net[in_res][k,:]
                    n=n+1
            res_new[k,:] = res_new[k,:]/n

        return res_new

def create_test_set_using_chimera(map_file, pdb_file, test_file):
    print "DEBUG 81"
    if os.path.exists(test_file):
        return
    bash_command = 'chimera-1.13 --nogui {} {} {} {}'.format(CHIMERA_SCRIPT,map_file, pdb_file, test_file)
    print "DEBUG 58"
    print bash_command

    process = subprocess.Popen(bash_command.split())
    output, error = process.communicate()
    return

def run_prediction(trained_net):
    N= 10**5

    models=[]
    for n_data in trained_net["nets"]:
        model  = n_data["model"]
        model.load_weights(n_data["weights_file"])
        models.append(model)

    cons = net1["consensus"]

    test_file_name = trained_net["test_boxes_file"]

    # load data - valid
    data_mtrx,filter_matrix,C,centers,labels = load_swap_labeled_5tuple_data(test_file_name)

    box_centers_ijk = get_box_centers_for_detection(data_mtrx,filter_matrix)
    box_centers_ijk_by_N = [box_centers_ijk[x:x+N] for x in range(0,len(box_centers_ijk),N)]

    res_data = np.zeros((0,21))
    box_centers = np.zeros((0,3))
    k=0
    for centers_ijk in box_centers_ijk_by_N:

        print "DEBUG ", k , "Start Collecting Boxes",time.ctime()
        pred_boxes, box_centers_xyz_N = get_boxes(data_mtrx,centers_ijk,C,normalization = Mean0Sig1Normalization)
        print "DEBUG ", k , "END Collecting Boxes",time.ctime()
        pred_features = np.reshape(pred_boxes,(len(pred_boxes),11,11,11,1))
        rslts = multimodelprediction(models,pred_features,cons)


        res_data = np.concatenate((res_data,rslts))
        box_centers = np.concatenate((box_centers,box_centers_xyz_N))
        print "DEBUG ", k , "END prediction Collecting Boxes",time.ctime()
        k+=1

    centers_arr = np.asarray(centers)
    det_res =  DetNetResults(res_folder =trained_net["res_folder"],labels_names=trained_net["labeling"].get_labels_to_names_dict(),xyz=box_centers,results=res_data,centers_labels=(centers_arr,labels),name = trained_net['name'])

    return det_res


def multimodelprediction(nets, x, consensus):
    resdata  = []
    for nn in nets:
        resdata.append(nn.predict(x))
    return consensus.calc_consensus(resdata)


NETS_FOLDER = dir_path +'/../../data/nets_data/'
BASE_FOLDER = NETS_FOLDER+'/det_analysis_2729/'
SUM_FOLDER =  BASE_FOLDER + '/sum_all_nets/'
TEST_PDB_FILE = dir_path +'/../../data//cryoEM/raw_data/res2729/3j9c_all.pdb'
TEST_REAL_MAP_FILE = dir_path +'/../../data//cryoEM/raw_data/res2729/emd-6224.map'
TEST_SIM_MAP_FILE = dir_path +'/../../data//cryoEM/raw_data/res2729_rot10_mrc/3j9c_all_rot0_res27apix9.mrc'


TEST_SIM_BOXES = BASE_FOLDER + 'emd_6224_pdb_3j9c_sim.pkl.gz'
network = all_nets.V5_no_reg()
TEST_REAL_BOXES = BASE_FOLDER + 'emd_6224_pdb_3j9c_real.pkl.gz'
network = all_nets.V5_no_reg()


net1={}
net1["name"] = "Train: Sim Rotamers, Test: Sim 6224"
net1["nets"] = []
net1["nets"].append({"model": network.get_compiled_net(), "weights_file":NETS_FOLDER+'/v4_db2729simcorners/weights_updated.h5'})
net1["consensus"] = MeanProbConsensus()
net1["res_folder"] = BASE_FOLDER+'/train_full_sim_test_sim/'
net1["test_pdb_file"] = TEST_PDB_FILE
net1["test_mrc_file"] = TEST_SIM_MAP_FILE
net1["test_boxes_file"] = TEST_SIM_BOXES
net1["labeling"] = LabelbyAAType
net1["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]


net2={}
net2["name"] = "Train: Sim Rotamers, Test: Real 6224"
net2["nets"] = []
net2["nets"].append({"model": network.get_compiled_net(), "weights_file":NETS_FOLDER+'/v4_db2729simcorners/weights_updated.h5'})
net2["consensus"] = MeanProbConsensus()
net2["res_folder"] = BASE_FOLDER+'/train_full_sim_test_real/'
net2["test_pdb_file"] = TEST_PDB_FILE
net2["test_mrc_file"] = TEST_REAL_MAP_FILE
net2["test_boxes_file"] = TEST_REAL_BOXES
net2["labeling"] = LabelbyAAType
net2["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]


net3={}
net3["name"] = "Train: Real Maps, Test: Real 6224"
net3["nets"] = []
net3["nets"].append({"model": network.get_compiled_net(), "weights_file":NETS_FOLDER+'/v4_db2729_real_rot10/weights_updated.h5'})
net3["consensus"] = MeanProbConsensus()
net3["res_folder"] = BASE_FOLDER+'/train_real_test_real/'
net3["test_pdb_file"] = TEST_PDB_FILE
net3["test_mrc_file"] = TEST_REAL_MAP_FILE
net3["test_boxes_file"] = TEST_REAL_BOXES
net3["labeling"] = LabelbyAAType
net3["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]


net4={}
net4["name"] = "Train: Real Maps+Sim Rotamers, Test: Real 6224"
net4["nets"] = []
net4["nets"].append({"model": network.get_compiled_net(), "weights_file": NETS_FOLDER+'/v4_db2729_real_rot10_plus_sim/weights_updated.h5'})
net4["consensus"] = MeanProbConsensus()
net4["res_folder"] = BASE_FOLDER+'/train_real_sim_test_real/'
net4["test_pdb_file"] = TEST_PDB_FILE
net4["test_mrc_file"] = TEST_REAL_MAP_FILE
net4["test_boxes_file"] = TEST_REAL_BOXES
net4["labeling"] = LabelbyAAType
net4["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]

net5={}
net5["name"] = "Mean of Three, Test: Real 6224"
net5["nets"] = []
net5["nets"].append(net2["nets"][0])
net5["nets"].append(net3["nets"][0])
net5["nets"].append(net4["nets"][0])
net5["consensus"] = MeanProbConsensus()
net5["res_folder"] = BASE_FOLDER+'/mean_of_three/'
net5["test_pdb_file"] = TEST_PDB_FILE
net5["test_mrc_file"] = TEST_REAL_MAP_FILE
net5["test_boxes_file"] = TEST_REAL_BOXES
net5["labeling"] = LabelbyAAType
net5["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]



net6={}
net6["name"] = "Mean of Real + Sim, Test: Real 6224"
net6["nets"] = []
net6["nets"].append(net2["nets"][0])
net6["nets"].append(net3["nets"][0])
net6["consensus"] = MeanProbConsensus()
net6["res_folder"] = BASE_FOLDER+'/mean_of_three/'
net6["test_pdb_file"] = TEST_PDB_FILE
net6["test_mrc_file"] = TEST_REAL_MAP_FILE
net6["test_boxes_file"] = TEST_REAL_BOXES
net6["labeling"] = LabelbyAAType
net6["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]

net7={}
net7["name"] = "Majority of Three, Test: Real 6224"
net7["nets"] = []
net7["nets"].append(net2["nets"][0])
net7["nets"].append(net3["nets"][0])
net7["nets"].append(net4["nets"][0])
net7["consensus"] = MajorityConsensus()
net7["res_folder"] = BASE_FOLDER+'/maj_of_three/'
net7["test_pdb_file"] = TEST_PDB_FILE
net7["test_mrc_file"] = TEST_REAL_MAP_FILE
net7["test_boxes_file"] = TEST_REAL_BOXES
net7["labeling"] = LabelbyAAType
net7["labels_to_print"] = [[x] for x in range(21)]+[[15,2,11,14]]

nets_trained=[net7,net1,net2,net3,net4,net5,net6]

for nn in nets_trained:
    create_test_set_using_chimera(nn["test_mrc_file"],nn["test_pdb_file"],nn["test_boxes_file"])


all_res = []

for nn in nets_trained:
    det_res = run_prediction(nn)
    all_res.append(det_res)

res_sum = DetRes4Nets(all_res,res_folder = SUM_FOLDER)

#res_sum.load_all_data()
#dtrs.create_over_all_text_res_file()
for x in range(1,21):
    res_sum.plot_results_per_label([x],N=1000)
    print res_sum.res_objects[0].labels_names[x]

res_sum.save_all_data()
