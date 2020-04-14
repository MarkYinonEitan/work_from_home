import os
import sys
from glob import glob1
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.models import model_from_json


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)

from resultsplots import SingleNetResults
from dbloader import  load_class_5tuple_data,DBLoader
from networkanalyser import NetworkAnalyser


def get_network_no_dropout():
    OPTIMIZER = 'adagrad'
    #define network
    ## the network ## the network
    model = Sequential()
    model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Conv3D(50, (2,2,2)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))

    model.add(Dense(21))
    model.add(Activation('softmax'))

    # load weights into new model
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model

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

'GENERAL PARAMETERS'
batch_span=range(100,1000,100)
data_folder = dir_path +'/../../data/'
res_folder_base = data_folder + '/nets_data/v4_db0023/'
net_with_drop = get_network_with_dropout()

"SIMULATION NO DROP OUT"
res_folder_sim_no_drop_out = res_folder_base+'/sim_no_drop/'
mini_batch_size = 600
res=3
apix = -0.1
N_valid = 100
N_epoch = 10
mini_epochs = 5

#print('create networkanalyser + # mini batch size')
net_anal = NetworkAnalyser(net_with_drop, res_folder_sim_no_drop_out)
net_anal.plot_time_vs_mini_batch_size(batch_span=batch_span,N_train = 10000)

SIM_DATABASE_FOLDER =  data_folder +'/rotamersdata/DBres2325norm01/'

# data base
data_loader = DBLoader(apix,res,data_folder=SIM_DATABASE_FOLDER)
all_pdbs = data_loader.get_all_pdbs()
np.random.shuffle(all_pdbs)
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

res_data_sim = []
for n_epoch in range(N_epoch):
    net_with_drop.fit(td_features,td_labels , validation_split = 0.1,epochs=mini_epochs, batch_size=mini_batch_size)
    res_data_sim.append(net_with_drop.predict(vd_features))
    # serialize model to JSON
    model_json = net_with_drop.to_json()
    fname = "mod" +str(n_epoch)
    with open(res_folder_sim_no_drop_out + fname +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net_with_drop.save_weights(res_folder_sim_no_drop_out + fname +".h5")
    print("Saved model to disk")




#parameters




assert 1<0, "Stop here"


mini_batch_size = 400




model.load_weights(WEIGHTS_INITIAL)


#print 'DEBUG 10', ' next lines works -commented with ### works ### '



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



data_folder = dir_path +'/../../data/cryoEM/DB0023class_rot10/'
nets_folder = dir_path +'/../../data/nets_data/'
res_folder = nets_folder+'/DB0023class_rot10_v3/'

train_emds = ['8194','3295']
valid_emds = ['2984']

train_data_files = []
for train_num in train_emds:
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot0.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot1.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot2.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot3.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot4.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot5.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot6.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot7.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot8.pkl.gz'.format(train_num))
    train_data_files = train_data_files+glob1(data_folder,'*{}_rot9.pkl.gz'.format(train_num))

valid_data_files = []
for valid_num in valid_emds:
    valid_data_files = valid_data_files+glob1(data_folder,'*{}*.pkl.gz'.format(valid_num))

WEIGHTS_INITIAL = nets_folder+'/DB0023classnoaug_v3/mod99.h5'
NETWORK_FILE = nets_folder+'/net_v3_tr2325_drop/mod19.json'
OPTIMIZER ='adagrad'
mini_batch_size = 500

N_epoch = 100
mini_epochs = 4

#load pretreained network
json_file = open(NETWORK_FILE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights(WEIGHTS_INITIAL)

# load data - valid
vd_boxes,vd_centers,vd_labels,_,_ = load_class_5tuple_data([data_folder+x for  x in valid_data_files])
vd_features = np.reshape(vd_boxes,(len(vd_boxes),11,11,11,1))
vd_labels_cat = to_categorical(vd_labels)

# load data - train
td_boxes,td_centers,td_labels,_,_ = load_class_5tuple_data([data_folder+x for  x in train_data_files])
td_features = np.reshape(td_boxes,(len(td_boxes),11,11,11,1))
td_labels_cat = to_categorical(td_labels)


#label statistics
ua,uc = np.unique(td_labels,return_counts=True)
train_data_stat={ua[x]:uc[x] for x in range(len(uc))}

res_data = []
res_data.append(model.predict(vd_features))

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
"GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
"LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
"SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

l2n = {v: k for k, v in label_dict.iteritems()}
l2n[5] = "CYS"
l2n[15] = "PRO"



for n_epoch in range(N_epoch):
    model.fit(td_features,td_labels_cat , validation_data = (vd_features,vd_labels_cat),epochs=mini_epochs, batch_size=mini_batch_size)
    res_data.append(model.predict(vd_features))
    # serialize model to JSON
    model_json = model.to_json()
    fname = "mod" +str(n_epoch)
    with open(res_folder + fname +".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(res_folder + fname +".h5")
    print("Saved model to disk")
    reslts = SingleNetResults(res_folder, l2n,res_per_epoch=res_data, valid_data=(vd_boxes,np.asarray(vd_labels)),train_data_stat = train_data_stat)
    reslts.calc_results()
    reslts.save_detection_graphs_one_run()

reslts.save_data()
