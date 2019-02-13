import os
import sys
from glob import glob1
import numpy as np

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.models import model_from_json
from keras import regularizers

import resultsplots
from resultsplots import SingleNetResults
import dbloader
from dbloader import  load_class_5tuple_data
import all_nets

data_folder = dir_path +'/../../data/cryoEM/DB0023class_rot10/'
nets_folder = dir_path +'/../../data/nets_data/'
res_folder = nets_folder+'/DB0023class_rot10_v3/'

train_emds = ['8762','8194','3295']
valid_emds = ['2984']



train_data_files = []
for train_num in train_emds:
    train_data_files = train_data_files+glob1(data_folder,'*{}*.pkl.gz'.format(train_num))

valid_data_files = []
for valid_num in valid_emds:
    valid_data_files = valid_data_files+glob1(data_folder,'*{}*.pkl.gz'.format(valid_num))

WEIGHTS_INITIAL = nets_folder+'/v4_db0023/sim_no_drop/mod7.h5'
NETWORK_FILE = nets_folder+'/v4_db0023/sim_no_drop/mod7.json'
ROOT_RES_FOLDER =  nets_folder+ '/v4_db0023_10_nets/'
OPTIMIZER ='adagrad'
mini_batch_size = 300

N_epoch = 50
mini_epochs = 4


#nets
nets_to_train = []
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net1/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net2/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net3/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net4/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net5/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
nets_to_train.append({"folder":ROOT_RES_FOLDER+'net6/', "resData":[],"net":all_nets.V5_no_reg().get_compiled_net()})
#
nets_results = []
nets_results.append({"folder":ROOT_RES_FOLDER+'nets_1_to_3/',"resData":[]})
nets_results.append({"folder":ROOT_RES_FOLDER+'nets_1_to_4/',"resData":[]})
nets_results.append({"folder":ROOT_RES_FOLDER+'nets_1_to_6/',"resData":[]})

#initial weights
#for nn in nets_to_train:
#    nn["net"].load_weights(WEIGHTS_INITIAL)


# load data - valid
valid_dict={}
dbloader.load_train_data_to_dict([data_folder+x for  x in valid_data_files],valid_dict)
vd_boxes = valid_dict["boxes"]
vd_labels = valid_dict["labels"]

vd_features = np.reshape(vd_boxes,(len(vd_boxes),11,11,11,1))
vd_labels_cat = to_categorical(vd_labels)

# load data - train
train_dict={}
dbloader.load_train_data_to_dict([data_folder+x for  x in train_data_files],train_dict)
td_boxes = train_dict["boxes"]
td_labels = train_dict["labels"]
td_features = np.reshape(td_boxes,(len(td_boxes),11,11,11,1))
td_labels_cat = to_categorical(td_labels)


#label statistics
ua,uc = np.unique(td_labels,return_counts=True)
train_data_stat={ua[x]:uc[x] for x in range(len(uc))}

class_weight = {x:float(sum(train_data_stat.values()))/train_data_stat[x] for x in train_data_stat.keys()}


label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

l2n = {v: k for k, v in label_dict.iteritems()}
l2n[5] = "CYS"
l2n[15] = "PRO"


for n_epoch in range(N_epoch):
    for net in nets_to_train:
        model = net["net"]
        res_folder = net["folder"]
        res_data =net["resData"]
        model.fit(td_features,td_labels_cat , validation_data = (vd_features,vd_labels_cat),epochs=mini_epochs, batch_size=mini_batch_size)
        res_data.append(model.predict(vd_features))
        # serialize weights to HDF5
        fname = "mod" +str(n_epoch)
        model.save_weights(res_folder + fname +".h5")
        reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=(vd_boxes,np.asarray(vd_labels)),train_data_stat = train_data_stat)
        reslts.calc_results()
        reslts.save_detection_graphs_one_run()

    res_1_to_3 = 0
    res_1_to_4 = 0
    res_1_to_6 = 0
    for k in range(3):
        res_1_to_3 = res_1_to_3+nets_to_train[k]["resData"][-1]
    for k in range(4):
        res_1_to_4 = res_1_to_6+nets_to_train[k]["resData"][-1]
    for k in range(6):
        res_1_to_6 = res_1_to_6+nets_to_train[k]["resData"][-1]
    res_1_to_3 = res_1_to_3/3
    res_1_to_4 = res_1_to_4/4
    res_1_to_6 = res_1_to_6/6

    res_data = nets_results[0]["resData"]
    res_folder = nets_results[0]["folder"]
    res_data.append(res_1_to_3)
    reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=(vd_boxes,np.asarray(vd_labels)),train_data_stat = train_data_stat)
    reslts.calc_results()
    reslts.save_detection_graphs_one_run()
    print("Saving 1_3 Net Results")
    reslts.save_data()

    res_data = nets_results[1]["resData"]
    res_folder = nets_results[1]["folder"]
    res_data.append(res_1_to_4)
    reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=(vd_boxes,np.asarray(vd_labels)),train_data_stat = train_data_stat)
    reslts.calc_results()
    reslts.save_detection_graphs_one_run()
    print("Saving 1_4 Net Results")
    reslts.save_data()

    res_data = nets_results[2]["resData"]
    res_folder = nets_results[2]["folder"]
    res_data.append(res_1_to_6)
    reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=(vd_boxes,np.asarray(vd_labels)),train_data_stat = train_data_stat)
    reslts.calc_results()
    reslts.save_detection_graphs_one_run()
    print("Saving 1_6 Net Results")
    reslts.save_data()
