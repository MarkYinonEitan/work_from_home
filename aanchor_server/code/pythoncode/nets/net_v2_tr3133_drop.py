import os
import sys
import time
import numpy as np
import random
import matplotlib.pyplot as plt

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


import networkanalyser
reload(networkanalyser)
from networkanalyser import NetworkAnalyser
import resultsplots
reload(resultsplots)
from resultsplots import SingleNetResults
import dbloader
reload(dbloader)
from dbloader import DBLoader
#import dbcreator
#reload(dbcreator)
#from dbcreator import LabelbyAAType

#parameters
batch_span=range(200,1000,20)
mini_batch_size = 400
res=3
apix = -0.1
N_valid = 100
N_epoch = 20
mini_epochs = 10
data_folder = dir_path +'/../../data/'
res_folder = data_folder + '/nets_data/net_v2_tr3133_drop/'

OPTIMIZER = 'adagrad'
WEIGHTS_INITIAL = data_folder + '/nets_data/net_v2_tr2931_drop/mod19.h5'
DATABASE_FOLDER =  data_folder +'/rotamersdata/DBres3133norm01/'



# data base
data_loader = DBLoader(apix,res,data_folder=DATABASE_FOLDER)

all_pdbs = data_loader.get_all_pdbs()
random.shuffle(all_pdbs)
N_pdbs = len(all_pdbs)
validation_pdbs = all_pdbs[-N_valid:]
train_per_miniepoch = [all_pdbs[:-N_valid]]
train_per_miniepoch = [train_per_miniepoch[0][0:4000]]

data_loader.set_valid_data(validation_pdbs)
data_loader.set_train_data(train_per_miniepoch)
## Data
train_data = data_loader.get_train_data(0)

td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
td_labels = to_categorical(train_data[1])

valid_data = data_loader.get_valid_data()
vd_features = np.reshape(valid_data[0],(len(valid_data[0]),11,11,11,1))
vd_labels = to_categorical(valid_data[1])


## the network ## the network
model = Sequential()
model.add(Conv3D(40, (2,2,2),input_shape=(11,11,11,1)))
model.add(Activation('relu'))
model.add(Conv3D(40, (2,2,2)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dense(21))
model.add(Activation('softmax'))

# load weights into new model
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights(WEIGHTS_INITIAL)

#print('create networkanalyser + # mini batch size')
#net_anal = NetworkAnalyser(model, res_folder)
#net_anal.plot_time_vs_mini_batch_size(batch_span=batch_span,N_train = 10000)

#print 'DEBUG 10', ' next lines works -commented with ### works ### '
res_data = []
for n_epoch in range(N_epoch):
    model.fit(td_features,td_labels , validation_split = 0.1,epochs=mini_epochs, batch_size=mini_batch_size)
    res_data.append(model.predict(vd_features))
    # serialize model to JSON
    model_json = model.to_json()
    fname = "mod" +str(n_epoch)
    with open(res_folder + fname +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(res_folder + fname +".h5")
    print("Saved model to disk")



#label statistics
ua,uc = np.unique(train_data[1],return_counts=True)
train_data_stat={ua[x]:uc[x] for x in range(len(uc))}


label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

l2n = {v: k for k, v in label_dict.iteritems()}
l2n[5] = "CYS"
l2n[15] = "PRO"

reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=valid_data,train_data_stat = train_data_stat)
reslts.save_data()
reslts.load_data()
reslts.save_detection_graphs_one_run()
