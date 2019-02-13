import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import gzip

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '../utils/'
sys.path.append(utils_path)

from process_rotamers_data import get_mrc_file_name, get_pdb_id
from process_rotamers_data import read_rotamers_data_text_file

def calc_detection(soft_max_probs):
    det = np.argmax(soft_max_probs,axis=1)
    return det

def calc_detection_accuracy(dets, y,labels ):
 
    N = np.array([sum(y == lb) for lb in labels ],dtype=float) +0.0001
    true_detections = dets == y
    n_det=np.array([sum((true_detections) & (y == lb)) for lb in labels])
    det_acc = []
    return n_det/N

def calc_det_vs_prob(lb,probs,y,prob_span = np.arange(0,1+0.1,0.1)):

    y_1 = y==lb
    y_0 = y!=lb

    probs_x = []
    det_y = []
    fa_y = []
    n_smpls = []
    for in_p in range(1,len(prob_span)):
        prob_start = prob_span[in_p-1]
        prob_end = prob_span[in_p]
        probs_x.append((prob_start+prob_end)/2)
        in_prob = (probs<=prob_end) & (probs>=prob_start)
        n_prob = np.sum(in_prob)+0.00001
        det_y.append(np.sum(y_1 & in_prob)/n_prob)
        fa_y.append(np.sum(y_0 & in_prob)/n_prob)
        n_smpls.append(n_prob)    
    
    return probs_x, det_y,fa_y

def calc_detection_matrix(dets,y,lbs):
    det_mat = np.zeros((len(lbs),len(lbs)),dtype=float)
    for lb_true in lbs:
        in_true = y == lb_true
        n_true = np.sum(in_true)+0.001
        for lb_det in lbs:
            in_det = dets==lb_det
            det_mat[lb_true,lb_det] = np.sum(in_det & in_true)/n_true
    return det_mat
             



def calc_FA_rate(dets, y,labels):
    """
    False Alarm Rate = m/n
    m - number of false positive detections
    n - number  of samples of other labels
    """
    N = np.array([sum(y != lb) for lb in labels ],dtype=float)+0.0001
    falses_detections = dets != y
    n_fa=np.array([sum((falses_detections) & (y == lb)) for lb in labels])
    return n_fa/N

def save_detection_martix(dets,y,labels_to_keys,filename):
    lbs = labels_to_keys.keys()
    names = [labels_to_keys[x] for x in lbs]
    A = calc_detection_matrix(dets,y,lbs)
    plt.close()
    plt.imshow(A,cmap = 'Paired')
    plt.colorbar()
    plt.xticks(lbs,names,rotation=90)
    plt.yticks(lbs,names)
    plt.title('A[I,J] = Prob(Detecting AA of type I, given AA of type J ]')
    ax = plt.gca()
    ax.set_xticks(np.array(lbs)+0.5, minor=True)
    ax.set_yticks(np.array(lbs)+0.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    plt.savefig(filename)

def detection_graphs_one_run(res_per_epoch, y, label_dict, folder_name ):
    labels_to_keys = {v: k for k, v in label_dict.iteritems()}
    labels = labels_to_keys.keys()
    #detections
    dets=[calc_detection(one_epoch_data) for one_epoch_data in res_per_epoch]
    det_rate=[calc_detection_accuracy(det_one_epoch,y,labels) for det_one_epoch in dets]
    fa_rate=[calc_FA_rate(det_one_epoch,y, labels) for det_one_epoch in dets]
    
    # detection matrix
    det_mat_filename = folder_name+'/det_mtrx.png'
    save_detection_martix(dets[-1],y,labels_to_keys,det_mat_filename)

    epoch_x_axis = range(len(res_per_epoch))
    for lb in labels:
        file_name = folder_name+'/res_'+labels_to_keys[lb] +'.png'
        det_rate_one_label = [x[lb]  for x in det_rate]
        fa_rate_one_label = [x[lb]  for x in fa_rate]

        plt.subplot(2,1,1)
        plt.title('Detection Accuracy' + labels_to_keys[lb])
        det_ln,=plt.plot(epoch_x_axis,det_rate_one_label,label='detection rate')
        fa_ln,=plt.plot(epoch_x_axis,fa_rate_one_label,label='FA rate')
        plt.xlabel('epoch')
        plt.legend(handles=[det_ln, fa_ln])
        plt.grid = True

        plt.subplot(2,1,2)
        plt.title('Last Epoch SoftMax Output (Probability) ' + labels_to_keys[lb])
        #calculate
        r_x, det_y, fa_y = calc_det_vs_prob(lb,res_per_epoch[-1][:,lb],y)
        det_ln,=plt.plot(r_x,det_y,label='Prob of True')
        fa_ln,=plt.plot(r_x,fa_y,label='Prob of False')
        plt.xlabel('Probability')
        plt.legend(handles=[det_ln, fa_ln])
        plt.grid = True
        plt.savefig(file_name)
        plt.close()


res_file_name = '/Users/macbookpro/Documents/rozSVN\/Projects/NNcryoEM/data/temp/res_for_plot.pkl.gz'

#load file
f = gzip.open(res_file_name, 'rb')
ws,res_per_epoch_conv,res_per_epoch, valid_data_y   = cPickle.load(f)
f.close()

label_dict = {"ALA":1,"ARG":2,"ASN":3,"ASP":4,"CYS":5,"CYH":5,"CYD":5,
    "GLN":6,"GLU":7,"GLY":8,"HIS":9,"ILE":10,
    "LEU":11,"LYS":12,"MET":13,"PHE":14,"PRO":15,"TPR":15,"CPR":15,
    "SER":16,"THR":17,"TRP":18,"TYR":19,"VAL":20,"NONE":0}

res_folder = '/Users/macbookpro/Documents/rozSVN\/Projects/NNcryoEM/data/temp/'
detection_graphs_one_run(res_per_epoch_conv, valid_data_y , label_dict, res_folder )

