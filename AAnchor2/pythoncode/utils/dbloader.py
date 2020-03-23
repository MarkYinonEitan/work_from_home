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


try:
  from scipy import ndimage
except ImportError:
  print ("RUN without scipy")

from process_rotamers_data import read_rotamers_data_text_file
from process_rotamers_data import get_mrc_file_name,get_pdb_id

def ijk_to_global_xyz(ijk,C):
    i = ijk[:,0]
    j = ijk[:,1]
    k = ijk[:,2]
    x = C[0]*i+C[1]*j+C[2]*k+C[3]
    y = C[4]*i+C[5]*j+C[6]*k+C[7]
    z = C[8]*i+C[9]*j+C[10]*k +C[11]

    xyz = np.vstack((x,y,z)).T
    return xyz

def global_xyz_to_ijk(xyz,C):

    C_m = np.asarray([[C[0],C[1],C[2]],[C[4],C[5],C[6]],[C[8],C[9],C[10]]])
    C_inv = np.linalg.inv(C_m)

    x0 = xyz[:,0]-C[3]
    y0 = xyz[:,1]-C[7]
    z0 = xyz[:,2]-C[11]
#    i = np.round(C_inv[0][0]*x0+C_inv[0][1]*y0+C_inv[0][2]*z0)
#    j = np.round(C_inv[1][0]*x0+C_inv[1][1]*y0+C_inv[1][2]*z0)
#    k = np.round(C_inv[2][0]*x0+C_inv[2][1]*y0+C_inv[2][2]*z0)
    i = C_inv[0][0]*x0+C_inv[0][1]*y0+C_inv[0][2]*z0
    j = C_inv[1][0]*x0+C_inv[1][1]*y0+C_inv[1][2]*z0
    k = C_inv[2][0]*x0+C_inv[2][1]*y0+C_inv[2][2]*z0

    ijk = np.vstack((i,j,k)).T
    return ijk


def box_from_box_center(x,y,z,cbs=11,apx=1):

    x_gr = np.linspace(x-cbs/2*apx,x+cbs/2*apx,cbs)
    y_gr = np.linspace(y-cbs/2*apx,y+cbs/2*apx,cbs)
    z_gr = np.linspace(z-cbs/2*apx,z+cbs/2*apx,cbs)
    X,Y,Z = np.meshgrid(x_gr,y_gr,z_gr,indexing = 'ij')
    box_xyz = (X.astype(int),Y.astype(int),Z.astype(int))

    return box_xyz


def filter_dy_mean(data_mtrx,cbs=11):
    assert cbs==11
    mean_all = np.mean(data_mtrx)
    mean_weights = np.ones((11,11,11))/(11*11*11)
    mean_matrix = ndimage.filters.convolve(data_mtrx,mean_weights,cval=-10.0**10,mode='constant',origin=(5,5,5))
    is_above_mean = mean_matrix>mean_all
    return is_above_mean


def get_box_centers_for_detection(data_mtrx,filter_matrix,cbs=11):
    assert cbs==11
    is_above_mean = filter_dy_mean(data_mtrx)
    box_origins = np.where(np.logical_and(is_above_mean,filter_matrix>0))
    box_centers_zipped = zip(box_origins[0]+5,box_origins[1]+5,box_origins[2]+5)
    return box_centers_zipped


#### Load the Data and Labels
def save_det_labeled_5tuple_data(filename,data_mtrx,filter_matrix,C,centers,labels):
    f = gzip.open(filename, "w")
    pickle.dump((data_mtrx,filter_matrix,C,centers,labels), f)
    f.close()
    return

def load_swap_labeled_5tuple_data(filename):
    f = gzip.open(filename, 'rb')
    mtrx_not_swapped,filter_matrix_not_swapped,C,centers,labels  = pickle.load(f,encoding="bytes")
    f.close()
    data_mtrx = np.swapaxes(mtrx_not_swapped,0,2)
    filter_matrix = np.swapaxes(filter_matrix_not_swapped,0,2)
    return data_mtrx,filter_matrix,C,centers,labels

def save_class_5tuple_data(filename,boxes,centers,labels,centers_pdb,labels_pdb):
    f = gzip.open(filename, "w")
    pickle.dump((boxes,centers,labels,centers_pdb,labels_pdb), f)
    f.close()
    return

def load_class_5tuple_data(f_names):
    boxes =[]
    centers=[]
    labels=[]
    centers_pdb=[]
    labels_pdb=[]
    k=0
    for filename in f_names:
        f = gzip.open(filename, 'rb')
        boxes_1,centers_1,labels_1,centers_pdb_1,labels_pdb_1  = pickle.load(f)
        f.close()
        boxes = boxes+ boxes_1
        centers = centers+centers_1
        labels= labels+labels_1
        centers_pdb = centers_pdb+centers_pdb_1
        labels_pdb = labels_pdb+labels_pdb_1
        print ("DEBUG 34: ",k, filename)
        k=k+1

    #convert labels to integers
    return boxes,centers,labels,centers_pdb,labels_pdb


class NoNormalization(object):
    @staticmethod
    def normamlize_3D_box( box):
        return box

class Mean0Sig1Normalization(object):
    @staticmethod
    def normamlize_3D_box(bx):
        bx_var = np.var(bx)
        #assert bx_var>0.001
        if bx_var<0.00000001:
            bx_norm = -999*np.ones(bx.shape)
        else:
            bx_norm = (bx-np.mean(bx))/np.sqrt(bx_var)
        return bx_norm


def get_boxes(data_mtrx,centers_ijk,C,normalization = Mean0Sig1Normalization,cbs=11):
    assert cbs ==11
    pos_reference = [100,100,100]
    bx_ref = box_from_box_center(pos_reference[0],pos_reference[1],pos_reference[2],cbs=cbs)

    pred_boxes=[]
    for ijk in centers_ijk:
        bx_indcs = (bx_ref[0]-pos_reference[0]+ijk[0],bx_ref[1]-pos_reference[1]+ijk[1],bx_ref[2]-pos_reference[2]+ijk[2])
        bx_indcs_int =(np.round(bx_indcs[0]).astype(int),np.round(bx_indcs[1]).astype(int),np.round(bx_indcs[2]).astype(int))
        k = np.where(np.abs(bx_indcs_int[0]-bx_indcs[0])>=0.01)
        box_no_norm = data_mtrx[bx_indcs_int[0],bx_indcs_int[1],bx_indcs_int[2]]
        pred_boxes.append(normalization.normamlize_3D_box(box_no_norm))

    centers = ijk_to_global_xyz(np.asarray([[x[0],x[1],x[2]] for x in centers_ijk]),C)

    return pred_boxes, centers

def get_all_pdbs(data_folder):
    all_pdbs =[get_pdb_id(x)  for x in glob.glob1(data_folder,'*.pkl.gz')]
    return all_pdbs

def load_train_data_to_dict(file_name_s, empty_dict,normalization = Mean0Sig1Normalization,cbs=11):
    start = timeit.default_timer()

    pred_boxes_all=[]
    labels_all = []
    for data_file in file_name_s:
        # load data - valid
        try:
            data_mtrx,filter_matrix,C,centers,labels = load_swap_labeled_5tuple_data(data_file)
        except:
            print ("DEBUG LOAD ", data_file , "UNLOADED")
            continue
        box_centers_ijk = global_xyz_to_ijk(np.asarray(centers),C)

        pred_boxes,_= get_boxes(data_mtrx,box_centers_ijk,C,normalization)
        pred_boxes_all = pred_boxes_all+pred_boxes
        labels_all = labels_all+labels

    empty_dict["boxes"] = pred_boxes_all
    empty_dict["labels"] = labels_all

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
