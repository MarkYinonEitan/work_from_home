"""network3.py
~~~~~~~~~~~~~~
Class to work with datasets
"""

#### Libraries
# Standard library
import pickle
import gzip
import os
import time
import threading
# Third-party libraries
import numpy as np
import glob
import timeit
import sys
import csv
import dataset_loader

try:
  from scipy import ndimage
except ImportError:
  print ("RUN without scipy")

from process_rotamers_data import read_rotamers_data_text_file
from process_rotamers_data import get_mrc_file_name,get_pdb_id

class NoNormalization(object):
    @staticmethod
    def normamlize_3D_box( box, mean=MEAN, sigma = SIGMA):
        return box

class Mean0Sig1Normalization(object):
    @staticmethod
    def normamlize_3D_box(bx, mean=0, sigma = 1):
        bx_var = np.var(bx)
        #assert bx_var>0.001
        if bx_var<0.00000001:
            bx_norm = -999*np.ones(bx.shape)
        else:
            bx_norm = (bx-np.mean(bx))/np.sqrt(bx_var)*sigma+mean
        return bx_norm

class MeanSigNormalization(object):
    @staticmethod
    def normamlize_3D_box(bx, mean = MEAN, sigma = SIGMA ):
        bx_var = np.var(bx)
        #assert bx_var>0.001
        if bx_var<0.00000001:
            bx_norm = -999*np.ones(bx.shape)
        else:
            bx_norm = (bx-np.mean(bx))/np.sqrt(bx_var)*sigma+mean
        return bx_norm


def get_all_pdbs(data_folder):
    all_pdbs =[get_pdb_id(x)  for x in glob.glob1(data_folder,'*.csv')]
    return all_pdbs


def save_label_data_to_csv(em_boxes, vx_boxes, labels_dict, file_name_pref):

    file_name_csv, file_name_map, file_name_vox = dataset_loader.data_file_name(file_name_pref)

    #save map boxes
    box_shape = boxes[0].shape
    all_boxes = [np.reshape(x,( box_shape[0]*box_shape[1]*box_shape[2])) for x in boxes]
    all_boxes_array = np.array(all_boxes)
    np.save(file_name_npy,all_boxes_array)

    #save vox boxes
    all_vxs=[]
    for vx_box in vx_boxes:
        vx_list = [vx_box[at_name] for at_name in ATOM_NAMES]
        all_vxs.append(np.concatenate(tuple(vx_list), axis = 3))
    all_vxs_array = np.array(all_vxs)
    np.save(file_name_vox,all_vxs_array)

    #save labels to csv
    csv_columns = list(labels_dict[0].keys())
    with open(file_name_csv, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in labels_dict:
            writer.writerow(data)
    return


def load_train_data_to_dict(file_name_s, empty_dict):

    number_fields = ["phi","box_center_y","chi2","chi3","chi1",\
    "box_center_x","pos","chi4","label","CG_pos_X","CG_pos_Y",\
    "CG_pos_Z","psi","box_center_z"]



    start = timeit.default_timer()

    empty_dict["boxes"] = []
    empty_dict["data"] =[]
    empty_dict["vx"] = []

    for file_name_pref in file_name_s:

        file_name_csv, file_name_map, file_name_vox = dataset_loader.data_file_name(file_name_pref)

        single_box_data = np.load(file_name_map)
        box_size = np.int(np.round(single_box_data.shape[1]**(1./3)))
        box_reshaped = [np.reshape(single_box_data[in_box,:], (box_size,box_size,box_size), order='C')  for in_box in range(single_box_data.shape[0])]
        empty_dict["boxes"] = empty_dict["boxes"] + box_reshaped

        single_vx_data = np.load(file_name_vox)
        vx_List = [single_vx_data[in_vx,:,:,:,:]  for in_vx in range(single_vx_data.shape[0])]
        empty_dict["vx"] = empty_dict["single_vx_data"] + vx_List


        with open(file_name_csv) as f_in:
                data_reader = csv.DictReader(f_in, delimiter=',')
                single_label_data = [x for x in data_reader]
                empty_dict["data"] = empty_dict["data"] + single_label_data

    for ky  in list(empty_dict["data"][0].keys()):
        if ky in number_fields:
            for row in empty_dict["data"]:
                row[ky] = np.float(row[ky])

    empty_dict["labels"] = [x["label"] for k in empty_dict["data"]]


    #add None Label
    empty_dict["boxes"].append(np.zeros(empty_dict["boxes"][-1].shape))
    empty_dict["labels"].append(0)


    stop = timeit.default_timer()

    print ("")
    print ("DATA LOADED  " +str(stop - start ) +"secs",time.ctime())
    print ("")

    return

class DBLoaderDetection(object):
    def __init__(self, data_folder=[], valid_data = [], train_data=[], pred_data=[]):
        if data_folder == []:
            if '__file__' in locals():
                dir_path = os.path.dirname(os.path.realpath(__file__))
            else:
                dir_path = os.getcwd()
            data_folder = dir_path+'/../../data/temp/'

        self.data_folder=data_folder
        self.valid_data_files = [data_folder+x for x in valid_data]
        self.train_data_files = [data_folder+x for x in train_data]
        self.pred_data_files = [data_folder+x for x in pred_data]
        self.valid_data = DBLoaderDetection.load_files_labeled(self.valid_data_files)
        self.train_data = DBLoaderDetection.load_files_labeled(self.train_data_files)
        self.pred_data = DBLoaderDetection.load_files_unlabeled(self.pred_data_files)

    @staticmethod
    def load_files_labeled(filenames):
        print ("DEBUG f",filenames)
        boxes = []
        centers=[]
        labels=[]
        for fname in filenames:
            f = gzip.open(fname, 'rb')
            boxes_1, centers_1,labels_1  = pickle.load(f)
            f.close()
            boxes = boxes+boxes_1
            centers = centers+centers_1
            labels = labels+labels_1
        return boxes, centers, np.asarray(labels)

    @staticmethod
    def load_files_unlabeled(filenames):
        boxes = []
        centers=[]
        for fname in filenames:
            f = gzip.open(fname[0], 'rb')
            boxes_1, centers_1  = pickle.load(f)
            f.close()
            boxes = boxes+boxes_1
            centers = centers+centers_1
        return boxes, centers
    def get_valid_data(self):
        return self.valid_data
    def get_train_data(self):
        return self.train_data
    def get_pred_data(self):
        return self.train_data




class DBLoader(object):

    def __init__(self, apix, res, data_folder=[]):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        if data_folder == []:
            if '__file__' in locals():
                dir_path = os.path.dirname(os.path.realpath(__file__))
            else:
                dir_path = os.getcwd()
            data_folder = dir_path+'/../../data/' + '/rotamersdata/'+ 'DB' + '_res' +str(int(res*10))+'apix'+str(int(apix*10))+'/'
        self.data_folder=data_folder
        self.apix = apix
        self.res = res
        self.pdbs_to_files = {}
        self.train_data_per_epoch = []
        self.valid_data = []
    @staticmethod
    def load_files_labeled(*args):
        labeled_data = args[0][0]
        for fname in args[1:]:
            f = gzip.open(fname[0], 'rb')
            data_one_file,  _ = pickle.load(f)
            f.close()
            labeled_data[0] = labeled_data[0]+data_one_file[0]
            labeled_data[1] = labeled_data[1]+data_one_file[1]
        return

    def get_train_data(self,mini_epoch,thread_num =0):
        pdbs = self.train_data_per_epoch[mini_epoch]
        f_names = [self.pdbs_to_files[x] for x in pdbs]
        if thread_num == 0:
            return self.load_data(f_names)
        else:
            return self.load_data_multithreads(f_names,thread_num = thread_num)

    def get_valid_data(self):
        return self.valid_data



    def get_all_filenames(self):
        return self.pdbs_to_files.values()


    def set_train_data(self, pdbs_mini_epoch):
        self.N_mini_epochs = len(pdbs_mini_epoch)
        self.train_data_per_epoch = pdbs_mini_epoch

    def set_valid_data(self,list_of_pdbs):
        valid_data_list = self.load_data([self.pdbs_to_files[x] for x in list_of_pdbs])
        self.valid_data = (np.asarray(valid_data_list[0]),np.asarray(valid_data_list[1]))



    #### Load the Data and Labels
    def load_data(self,filenames):
        labeled_data = [[],[]] # [ [data
        for file_name in  filenames:
            f = gzip.open(self.data_folder+file_name, 'rb')
            labeled_data_one_file,  _ = pickle.load(f)
            f.close()
            labeled_data[0] = labeled_data[0]+labeled_data_one_file[0]
            labeled_data[1] = labeled_data[1]+labeled_data_one_file[1]
        return labeled_data


    #### Load the Data and Labels
    def load_data_multithreads(self,filenames,thread_num = 1):

        #run in threads
        n_files_per_threads = len(filenames)/thread_num+1
        f_names_threads = [filenames[x:x+n_files_per_threads] for x in range(0,len(filenames),n_files_per_threads)]


        threads=[]
        all_data = []
        k=0
        for f_names_list in f_names_threads:
            var_name = 'labeled_data_' + str(k)
            exec(var_name + '=[[],[]]')
            exec('all_data.append(' + var_name + ')')

            full_path_names =[self.data_folder+f  for f in f_names_list]
            exec('thread_args = zip([' + var_name + ']+full_path_names)')
            t = threading.Thread(target=DBLoader.load_files_labeled, args=thread_args)
            threads.append(t)
            #t.start()
            k=k+1

        for t in threads:
            t.start()
        for t in threads:
        	t.join()

        labeled_data = [[],[]]
        for data in  all_data:
            labeled_data[0] = labeled_data[0]+data[0]
            labeled_data[1] = labeled_data[1]+data[1]

        return labeled_data




class LabelbyAAType(object):

    label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
    "GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
    "LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
    "SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

    @staticmethod
    def calc_label(res_data):
        """return the label assosiated with current entry in the rotamers data base
        """
        return LabelbyAAType.label_dict.get(res_data["Type"],-1)

    @staticmethod
    def calc_label_from(type):
        """return the label assosiated with current entry in the rotamers data base
        """
        return LabelbyAAType.label_dict.get(type,-1)


    @staticmethod
    def print_labels(file_name):
        """prints labels to a text file"""
        text_file = open(file_name, "w")
        for ky in LabelbyAAType.label_dict.keys():
            text_file.write(str(ky)+' : ' + str(LabelbyAAType.label_dict[ky])+"\n")
        text_file.close()


    @staticmethod
    def get_labels_to_names_dict():
        l2n = {v: k for k, v in LabelbyAAType.label_dict.items()}
        l2n[5] = "CYS"
        l2n[15] = "PRO"
        return l2n
