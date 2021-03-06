
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
class_weight= None):
    start = timeit.default_timer()

    td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
    td_labels = to_categorical(train_data[1])

    print ("DEBUG 8", td_features.shape, td_labels.shape)

    vd_features = np.reshape(valid_data[0],(len(valid_data[0]),11,11,11,1))
    vd_labels = to_categorical(valid_data[1])

    model.fit(td_features,td_labels , validation_data = (vd_features,vd_labels),epochs=mini_epochs, shuffle = True, batch_size=mini_batch_size, class_weight=class_weight)
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
    rsobjct.calc_results()
    rsobjct.save_data()
    rsobjct.save_detection_graphs_one_run()

    stop = timeit.default_timer()

    print ("")
    print ("RESULTS SAVED " +str(stop - start ) +"secs", time.ctime())
    print ("")

    return



class NetworkAnalyser:
    def __init__(self, net, res_folder=temp_folder, train_files=[],valid_files=[],initial_weight_file='nothing',mini_batch_size=100,max_valid_res = 1000000):

        self.net = net
        self.res_folder = res_folder
        self.train_files= train_files
        self.valid_files= valid_files
        self.initial_weight_file = initial_weight_file
        self.mini_batch_size=mini_batch_size
        self.max_valid_res = max_valid_res
        return

    def train_network(self, N_epoch = 50):

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
        dbloader.load_train_data_to_dict(self.train_files,train_load_dict)
        train_data =  (train_load_dict["boxes"],train_load_dict["labels"])
        td_features = np.reshape(train_data[0],(len(train_data[0]),11,11,11,1))
        td_labels = to_categorical(train_data[1])
        print ("Train set loaded", time.ctime())


        res_data =[]
        #load firts train and valid set
        label_class = dbloader.LabelbyAAType()
        reslts_obj = SingleNetResults(self.res_folder, label_class.get_labels_to_names_dict(), valid_data = valid_data)

        for epoch in range(N_epoch):
            res_dict = {'res':None}
            run_miniepochs(model, train_data, valid_data,5, self.mini_batch_size, res_dict, class_weight= self.net.get_class_weights())
            print ("EPOCH",epoch , "of",N_epoch)

            if N_epoch%10 ==0:
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
