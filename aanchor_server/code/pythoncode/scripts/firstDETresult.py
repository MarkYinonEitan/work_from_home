import os
import sys
import time
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
import dbloader
reload(dbloader)
from dbloader import  Mean0Sig1Normalization,load_swap_labeled_5tuple_data,get_box_centers_for_detection,get_boxes
import resultsplots
reload(resultsplots)
from resultsplots import DetNetResults



def get_network_with_dropout():
    OPTIMIZER = 'adagrad'
    #define network
    ## the network ## the network
    model = Sequential()
    model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv3D(50, (2,2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))

    model.add(Dense(21))
    model.add(Activation('softmax'))

    # load weights into new model
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model

def multimodelprediction(nets, x):
    resdata  = []
    for nn in nets:
        resdata.append(nn["net"].predict(pred_features))
    return sum(resdata)/len(resdata)



data_folder = dir_path +'/../../data/temp/'
nets_folder = dir_path +'/../../data/nets_data/'
res_folder = dir_path +'/../../data/temp/'
ROOT_RES_FOLDER =  nets_folder+ '/v4_db0023_10_nets/'


#nets
nets_trained = []
nets_trained.append({"folder":ROOT_RES_FOLDER+'net1/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})
nets_trained.append({"folder":ROOT_RES_FOLDER+'net2/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})
nets_trained.append({"folder":ROOT_RES_FOLDER+'net3/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})
nets_trained.append({"folder":ROOT_RES_FOLDER+'net4/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})
nets_trained.append({"folder":ROOT_RES_FOLDER+'net5/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})
nets_trained.append({"folder":ROOT_RES_FOLDER+'net6/', "resData":[],"net":get_network_with_dropout(),"weights_file":'mod19.h5'})


for nn in nets_trained:
    nn["net"].load_weights(nn["folder"]+nn["weights_file"])


normalization=Mean0Sig1Normalization
DATA_FILE = data_folder +'det2984.pkl.gz'
BOX_SIZE=11
N= 10**5



# load data - valid
data_mtrx,filter_matrix,C,centers,labels = load_swap_labeled_5tuple_data(DATA_FILE)

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
    rslts = multimodelprediction(nets_trained, pred_features)


    res_data = np.concatenate((res_data,rslts))
    box_centers = np.concatenate((box_centers,box_centers_xyz_N))
    print "DEBUG ", k , "END prediction Collecting Boxes",time.ctime()
    k+=1


label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

l2n = {v: k for k, v in label_dict.iteritems()}
l2n[5] = "CYS"
l2n[15] = "PRO"

centers_arr = np.asarray(centers)
det_res =  DetNetResults(res_folder =res_folder,labels_names=l2n,xyz=box_centers,results=res_data,centers_labels=(centers_arr,labels))
det_res.save_data()
det_res.load_data()
det_res.create_over_all_text_res_file()
for x in range(1,21):
    det_res.results_per_label([x],N=100)
    print det_res.labels_names[x]
