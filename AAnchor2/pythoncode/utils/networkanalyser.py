
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import sys
import timeit
import threading
import tensorflow as tf

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
nets_path = dir_path + '/../nets/'
sys.path.append(utils_path)
sys.path.append(nets_path)
import dbloader
from dbloader import LabelbyAAType
from resultsplots import SingleNetResults

temp_folder = dir_path +'/../data/temp/'

def run_miniepochs(model, train_data, valid_data,mini_epochs, mini_batch_size, results_dict,
class_weight= None, tensorboard_callbacks = []):
    start = timeit.default_timer()

    td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
    td_labels = to_categorical(train_data[1])

    print ("DEBUG 8", td_features.shape, td_labels.shape)

    vd_features = np.reshape(valid_data[0],(len(valid_data[0]),11,11,11,1))
    vd_labels = to_categorical(valid_data[1])

    model.fit(td_features,td_labels , validation_data = (vd_features,vd_labels),epochs=mini_epochs, batch_size=mini_batch_size, class_weight=class_weight , callbacks = tensorboard_callbacks)
    rslts = model.predict(vd_features)

    results_dict['res'] = rslts

    stop = timeit.default_timer()
    print ("")
    print ("EPOCH FINISHED " +str(stop - start ) +"secs",time.ctime())
    print ("")

    return

def save_results(res_data,train_data,rsobjct):
    start = timeit.default_timer()

    rsobjct.res_per_epoch = res_data[-10:]
    #label statistics
    ua,uc = np.unique(train_data[1],return_counts=True)
    for x in range(len(uc)) :
        ua[x] = round(ua[x])
        rsobjct.train_data_stat[ua[x]] = rsobjct.train_data_stat.get(ua[x],0) +uc[x]
    print("DEBUG 2324", res_data)
    rsobjct.calc_results()
    rsobjct.save_data()
    rsobjct.save_detection_graphs_one_run()

    stop = timeit.default_timer()

    print ("")
    print ("RESULTS SAVED " +str(stop - start ) +"secs", time.ctime())
    print ("")

    return


def load_train_in_parallel(train_file_names=None,train_data = None,valid_data=None,res_data =None, model=None,mini_epochs=5, mini_batch_size = 100, rsobjct=None):

    thrds = []

    res_dict = {'res':None}
    train_load_dict = {"boxes":None,"labels":None}

    if model !=None:
        thr_train = threading.Thread(target = run_miniepochs, args=(model, train_data, valid_data, mini_epochs, mini_batch_size ,res_dict))
        thrds.append(thr_train)

    if rsobjct!=None and len(res_data)>1 :
        thr_save_res = threading.Thread(target = save_results, args=(res_data,train_data,rsobjct))
        thrds.append(thr_save_res)

    if train_file_names!=None:
        thr_load = threading.Thread(target = dbloader.load_train_data_to_dict, args=(train_file_names,train_load_dict))
        thrds.append(thr_load)

    for t in thrds:
        t.start()
    for t in thrds:
        t.join()

    res_data.append(res_dict['res'])
    new_train_data = (train_load_dict["boxes"],train_load_dict["labels"])


    return new_train_data, res_data


class NetworkAnalyser:
    def __init__(self, net, res_folder=temp_folder, train_files=[],valid_files=[],initial_weight_file='nothing',mini_batch_size=100,max_valid_res = 1000000,labeling = LabelbyAAType):

        self.net = net
        self.res_folder = res_folder
        self.train_files= train_files
        self.valid_files= valid_files
        self.initial_weight_file = initial_weight_file
        self.mini_batch_size=mini_batch_size
        self.max_valid_res = max_valid_res
        self.labeling = labeling
        return

    def train_network_par(self, N_epoch = 10, mini_epochs=5,pdbs_per_mini_epoch=200):
        random.shuffle(self.train_files)
        train_files_per_miniepoch = [self.train_files[x:x+pdbs_per_mini_epoch] for x in range(0,len(self.train_files),pdbs_per_mini_epoch)]

        print("DEBUG 23",self.train_files )
        print("DEBUG 121223",train_files_per_miniepoch )

        #load firts train and valid set
        train_load_dict = {}
        dbloader.load_train_data_to_dict(train_files_per_miniepoch[0],train_load_dict)
        train_data =  (train_load_dict["boxes"],train_load_dict["labels"])
        td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
        td_labels = to_categorical(train_data[1])
        print ("First train set loaded", time.ctime())

        valid_load_dict = {}
        dbloader.load_train_data_to_dict(self.valid_files,valid_load_dict)
        valid_data = (valid_load_dict["boxes"][:self.max_valid_res],valid_load_dict["labels"][:self.max_valid_res])
        print ("Valid train loaded",time.ctime())

        #label statistics
        ua,uc = np.unique(train_data[1],return_counts=True)
        train_data_stat={ua[x]:uc[x] for x in range(len(uc))}

        res_data =[]
        reslts = SingleNetResults(self.res_folder, self.labeling.get_labels_to_names_dict(),res_per_epoch=res_data, valid_data=valid_data,train_data_stat = train_data_stat)

        model = self.net.get_compiled_net()
        if os.path.exists(self.initial_weight_file):
            model.load_weights(self.initial_weight_file)

        print ("DEBUG First Run")
        run_miniepochs(model, train_data, valid_data,mini_epochs, self.mini_batch_size, {})

        for epoch in range(N_epoch):
            random.shuffle(self.train_files)
            train_files_per_miniepoch = [self.train_files[x:x+pdbs_per_mini_epoch] for x in range(0,len(self.train_files),pdbs_per_mini_epoch)]

            init_weights = model.get_weights()
            upd_weights = []
            n_mini = 0;
            for  train_files in train_files_per_miniepoch[0:]:
                model.set_weights(init_weights)
                new_train_data, new_res_data = load_train_in_parallel(train_file_names= train_files,train_data = train_data,valid_data=valid_data,res_data =res_data, model=model,mini_epochs=mini_epochs, mini_batch_size = self.mini_batch_size,rsobjct=reslts)
                train_data = new_train_data
                res_data = new_res_data
                upd_weights.append(model.get_weights())
                print ("EPOCH: MINIEPOCH of ALL",epoch,n_mini, "of",len(train_files_per_miniepoch))
                n_mini +=1

            #avereging weights
            new_weights = list()
            for lr_weight in zip(*upd_weights):
                new_weights.append(np.mean(lr_weight,axis=0))
            model.set_weights(new_weights)

                # serialize weights to HDF5
            fname = "weights_updated"
            model.save_weights(reslts.res_folder + fname +".h5")


        return


    def train_network(self, N_epoch = 10, mini_epochs=5,pdbs_per_mini_epoch=200):

        model = self.net.get_compiled_net()
        if os.path.exists(self.initial_weight_file):
            model.load_weights(self.initial_weight_file)
            print("WEIGHTS LOADED")
        print ("Model Initiated",time.ctime())


        valid_load_dict = {}
        dbloader.load_train_data_to_dict(self.valid_files,valid_load_dict)
        valid_data = (valid_load_dict["boxes"][:self.max_valid_res],valid_load_dict["labels"][:self.max_valid_res])
        print ("Valid train loaded",time.ctime())



        train_load_dict = {}
        res_data =[]
        #load firts train and valid set
        reslts_obj = SingleNetResults(self.res_folder, self.labeling.get_labels_to_names_dict(), valid_data = valid_data)



        for epoch in range(N_epoch):
            #make list of all pdbs
            pdb_files_random = [random.choice(self.train_files) for x in range(pdbs_per_mini_epoch*mini_epochs)]
            train_files_per_miniepoch = [pdb_files_random[x:x+pdbs_per_mini_epoch] for x in range(0,pdbs_per_mini_epoch*mini_epochs,pdbs_per_mini_epoch)]

            for n_mini in range(mini_epochs):
                dbloader.load_train_data_to_dict(train_files_per_miniepoch[n_mini],train_load_dict)
                train_data =  (train_load_dict["boxes"],train_load_dict["labels"])
                td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
                td_labels = to_categorical(train_data[1])
                print ("Train set loaded", time.ctime())
                #label statistics
                res_dict = {'res':None}
                run_miniepochs(model, train_data, valid_data,mini_epochs, self.mini_batch_size, res_dict, class_weight= self.net.get_class_weights(), callbacks =tensorboard_callbacks)
                print ("EPOCH: MINIEPOCH of ALL",epoch,n_mini, "of",len(train_files_per_miniepoch))
                res_data.append(res_dict['res'])

            save_results(res_data,train_data,reslts_obj)
            model.save_weights(reslts_obj.res_folder + "weights_updated" +'_'+str(epoch)+".h5")

        return

    def plot_time_vs_mini_batch_size(self, batch_span = range(1,100,10), N_train=10000,
    save_or_show= 'save'):
        file_name = 'run_time_minibatch.png'
        #create train , test, valid set
        N_epochs = 10
        N_norm = 1000000

        #create database
        input_shape = self.net.layers[0].input_shape
        output_shape = self.net.layers[-1].output_shape
        n_in = np.prod(input_shape[1:])
        n_out = output_shape[1]
        train_x = np.random.random([N_train,n_in])
        train_y = np.random.choice(range(n_out),N_train)
        td_features  = np.reshape(train_x,(N_train,input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
        td_labels = to_categorical(train_y)
        batch_span_for_plot = []
        time_span = []

        for mini_batch_size in  batch_span:
            start_time = time.time()
            self.net.fit(td_features,td_labels , epochs=N_epochs, batch_size=mini_batch_size)
            end_time = time.time()
            run_time = end_time -start_time
            time_span.append(run_time/N_train/N_epochs*N_norm)
            batch_span_for_plot.append(mini_batch_size)

	        #plot or show
            plt.close()
            plt.plot(batch_span_for_plot,time_span)
            plt.xlabel('mini batch size')
            plt.ylabel('time for 10^6 samples [sec] ')
            plt.title('Train Time vs mini batch size')
            if save_or_show == 'save':
                plt.savefig(self.res_folder+file_name)
            else:
                plt.show()
