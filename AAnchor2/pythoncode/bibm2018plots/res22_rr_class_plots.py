import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import gzip
from scipy.spatial import KDTree

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)


import resultsplots
from resultsplots import SingleNetResults
from dbloader import LabelbyAAType



num_of_rotamers = {"ALA":1,"ARG":81,"ASN":18,"ASP":18,"CYS":3,"CYH":3,"CYD":3,
    "GLN":54,"GLU":54,"GLY":1,"HIS":18,"ILE":9,
    "LEU":9,"LYS":81,"MET":27,"PHE":12,"PRO":2,"TPR":2,"CPR":2,
    "SER":3,"THR":3,"TRP":9,"TYR":12,"VAL":3,"NONE":1}

labels_to_show =[ 6,  3, 18, 10,  9, 17,  5,  2, 16, 14,  4, 13,  7, 15, 19, 12, 20, 1,  8, 11]

#num_of_rotamers = {"ALA":1,"ARG":81,"ASN":3,"ASP":3,"CYS":3,"CYH":3,"CYD":3,"GLN":9,"GLU":9,"GLY":1,"HIS":3,"ILE":9,    "LEU":9,"LYS":81,"MET":27,"PHE":3,"PRO":1,"TPR":2,"CPR":2,    "SER":3,"THR":3,"TRP":9,"TYR":3,"VAL":3,"NONE":1}


def plot_reliability_curve(prob_vs_output,lgnd,filename):
    plt.close()
    plt.title('reliability plot: '+lgnd)
    x = np.asarray(prob_vs_output['prob_x'])
    y = np.asarray(prob_vs_output['det_y'])
    in_n0 = y>0.05

    plt.plot(x[in_n0] ,y[in_n0], 'C2-o')
    plt.plot([0,1] ,[0,1],'k-')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.axes().set_aspect(0.5)
    plt.savefig(filename)
    plt.close()
    return



def plot_total_accuracy(acc, lbs_names,lbs_order,title,filename):
    acc_to_plot = [acc[x] for x in lbs_order]


    x_ticks_pos = range(len(lbs_order))
    x_ticks_name= [lbs_names[lbs_order[x]] for x in x_ticks_pos]

    plt.close()
    plt.plot(x_ticks_pos,acc_to_plot,'C2-o')

    #for x in x_ticks_pos:
    #    plt.text(x, acc[lbs_order[x]], x_ticks_name[x], fontsize=10)

    plt.xticks(x_ticks_pos,x_ticks_name,rotation=90)
    plt.yticks(np.arange(0.0,1.0,0.1))
    plt.ylabel('Total Accuracy')
    plt.grid(True)
    plt.title(title)
    plt.savefig(filename)


def plot_det_vs_datasize(det_acc,train_data_stat,labels_names, title, filename):
    plt.close()
    labels = list(labels_names.keys())
    train_data_size = [train_data_stat.get(x,0) for x in labels]

    plt.plot(train_data_size[1:],det_acc[1:],marker = 's',linestyle = 'None')
    plt.xscale('log')
    plt.xlabel('Num of samples')
    plt.ylabel('Total Accuracy')
    plt.xticks([1e5,1e6,1e7])


    for x in labels:
        if x==0:
            continue
        plt.text(train_data_size[x], det_acc[x], labels_names[x], fontsize=8)
    plt.title(title)
    plt.savefig(filename)

def plot_det_vs_datasize_norm(det_acc,train_data_stat,labels_names, title, filename):
    plt.close()
    labels = list(labels_names.keys())
    train_data_size = [train_data_stat.get(x,0)/num_of_rotamers[labels_names[x]] for x in labels]

    plt.plot(train_data_size[1:],det_acc[1:],marker = 's',linestyle = 'None')
    plt.xscale('log')
    plt.xlabel('Num of samples / Num of Rotamers')
    plt.ylabel('Total Accuracy')
    plt.xticks([1e4,1e5,1e6])

    for x in labels:
        if x==0:
            continue
        plt.text(train_data_size[x], det_acc[x], labels_names[x], fontsize=8)
    plt.title(title)
    plt.savefig(filename)



def plot_detection_matrix(mtrx, title, labels, names, filename):
    plt.close()
    plt.imshow(mtrx,interpolation = 'none',cmap = 'nipy_spectral')
    plt.clim(0,1.0)
    plt.colorbar()
    plt.xticks(labels,names,rotation=90)
    plt.yticks(labels,names)
    plt.title(title)
    ax = plt.gca()
    ax.set_xticks(np.array(labels)+0.5, minor=True)
    ax.set_yticks(np.array(labels)+0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.savefig(filename)
    return



RES_FOLDER = dir_path + '/../../rep/bibm2018pics/'
NETS_FOLDER = dir_path + '/../../data/nets_data/'

RES = SingleNetResults(NETS_FOLDER+'/v4_db0023_real_rot10/', LabelbyAAType.get_labels_to_names_dict())
RES.load_data()
plot_detection_matrix(RES.det_matrix, '',RES.labels, RES.names, RES_FOLDER+'CM_22_RR.png')

plot_det_vs_datasize(RES.det_acc_per_epoch[-1],RES.train_data_stat, RES.labels_names, '',  RES_FOLDER+'DS_22_RR.png')
plot_det_vs_datasize_norm(RES.det_acc_per_epoch[-1],RES.train_data_stat, RES.labels_names, '',  RES_FOLDER+'DS_norm_22_RR.png')

plot_total_accuracy(RES.det_acc_per_epoch[-1], RES.labels_names,labels_to_show,'Res: $2.2$,  Train:EMDs 8194,3295,8762, Test: $\\beta$-galactosidase,EMD-2984. ',RES_FOLDER+'rr22_acc.png')


#plot_reliability_curve(RES.prob_vs_output[1],'ALA',RES_FOLDER+'rel_ALA_22_SS.png')
#plot_reliability_curve(RES.prob_vs_output[2],'ARG',RES_FOLDER+'rel_ARG_22_SS.png')
#plot_reliability_curve(RES.prob_vs_output[15],'PRO',RES_FOLDER+'rel_PRO_22_SS.png')
#plot_reliability_curve(RES.prob_vs_output[11],'LEU',RES_FOLDER+'rel_LEU_22_SS.png')
