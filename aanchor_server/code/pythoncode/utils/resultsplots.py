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
utils_path = dir_path + '../utils/'
sys.path.append(utils_path)

from process_rotamers_data import get_mrc_file_name, get_pdb_id
from process_rotamers_data import read_rotamers_data_text_file

import dbloader

TEMP_FOLDER =  dir_path +'/../../data/temp/'

class DetNetResults(object):
    def __init__(self, res_folder = TEMP_FOLDER, labels_names=[],xyz=[],results=[],centers_labels=([],[]),dist_thr = 2.0,filt_thr = 2.0,file_name = 'res_detection.pkl.gz',pdb_file = 'res.pdb',name = 'default'):
        self.res_folder = res_folder
        self.labels_names = labels_names
        self.xyz = xyz
        self.results = results
        self.centers_labels = centers_labels
        self.dist_thr = dist_thr
        self.filt_thr = filt_thr
        self.file_name = file_name
        self.pdb_file = pdb_file
        self.name = name
        return

    def save_data(self):
        f = gzip.open(self.res_folder+self.file_name, "w")
        cPickle.dump((self.labels_names,self.xyz,self.results,self.centers_labels), f)
        f.close()
        return

    def filter_detections(self, probs, dets, xyz,probs_all, dets_all, xyz_all):

        def is_good(xyz,lb,prob,xyz_tree,dets,probs):



            in_nearby = xyz_tree.query_ball_point(xyz,self.filt_thr)
            if len(in_nearby) == 0:
                5#return True

            lb_nearby = np.asarray([dets[x] for x in in_nearby])
            if (lb_nearby!=lb).any():
                5#return False
            if  np.sum(lb_nearby==lb)<3:
                5#return False
            probs_nearby = np.asarray([probs[x] for x in in_nearby])
            if (prob>=probs_nearby).all():
                return True

            return False

        # create  kdtree
        if (len(xyz_all)==0):
            probs_good = []
            dets_good = []
            xyz_good = []

            return probs_good, dets_good, xyz_good

        xyz_tree = KDTree(xyz_all)
        in_good = [is_good(xyz[k,:],dets[k],probs[k],xyz_tree,dets_all,probs_all) for k in range(len(dets))]

        probs_good = probs[in_good]
        dets_good = dets[in_good]
        xyz_good = xyz[in_good,:]

        return probs_good,dets_good,xyz_good

    def write_det_to_pdb_file(self, dets,xyz,filename = None):
        if filename ==None:
            filename = self.pdb_file
        string_line = "HETATM AAAA  K   RES ABBBB     XXXXXXX YYYYYYY ZZZZZZZ  0.25 14.49           K  "

        f = open(self.res_folder+filename, 'w')

        for in_det in range(len(dets)):
            s = str(string_line)
            #position
            s=s.replace('XXXXXXX', '{:3.4g}'.format(xyz[in_det,0]).ljust(7))
            s=s.replace('YYYYYYY', '{:3.4g}'.format(xyz[in_det,1]).ljust(7))
            s=s.replace('ZZZZZZZ', '{:3.4g}'.format(xyz[in_det,2]).ljust(7))
            s=s.replace('RES', self.labels_names[dets[in_det]].ljust(3))
            s=s.replace('AAAA', str(in_det).ljust(4))
            s=s.replace('BBBB', str(in_det).ljust(4))
            f.write(s)
            f.write("\n")

        f.close()

        return

    def calc_results_for_labels(self,lbs,N=1000):
        def filter_and_sort_label(probs,xyz,dets,lbs,N):
            assert isinstance(dets,np.ndarray)
            assert isinstance(xyz,np.ndarray)
            assert isinstance(probs,np.ndarray)

            #filter
            in_det = np.in1d(dets,lbs)
            probs = probs[in_det]
            xyz = xyz[in_det,:]
            dets = dets[in_det]

            #sort
            in_sort = np.argsort(probs)[::-1]
            in_sort = in_sort[:N]
            probs = probs[in_sort]
            dets = dets[in_sort]
            xyz = xyz[in_sort,:]
            return probs, xyz,dets


        #get detections
        dets_all,xyz_all,probs_all = self.get_non_zero_preds()

        if len(probs_all)==0:
            RES = {}
            RES["n_reported"] = 0
            RES["is_false"] = []
            RES["is_true"] = []
            RES["prob_of_true"] = []
            RES["probs_f"] = []
            RES["dets_f"] = []
            RES["xyz_f"] = []

            return RES

        probs_lb,xyz_lb, dets_lb= filter_and_sort_label(probs_all,xyz_all,dets_all,lbs,N)
        #filter detections
        probs_f,dets_f,xyz_f =self.filter_detections(probs_lb, dets_lb, xyz_lb,probs_all, dets_all, xyz_all)
        # calc references
        ref_labels, dist_to_reference = self.calc_referenced_labels(xyz_f,self.centers_labels[0],self.centers_labels[1])
        #FA
        false_dets = np.not_equal(ref_labels,dets_f)
        true_dets = np.equal(ref_labels,dets_f)
        n_reported = np.asarray(range(len(ref_labels)),dtype = float)+1
        prob_of_true = np.cumsum(true_dets)/n_reported

        RES={}
        RES["n_reported"] = n_reported
        RES["is_false"] = false_dets
        RES["is_true"] = true_dets
        RES["prob_of_true"] = prob_of_true
        RES["probs_f"] = probs_f
        RES["dets_f"] = dets_f
        RES["xyz_f"] = xyz_f

        return RES

    def plot_results_per_label(self,lbs,N=1000):

        lb_name = [self.labels_names[lb] + '_' for lb in lbs]
        # filenames
        res_text_file = '/res_{}_data.txt'.format(lb_name)
        res_pdb_file = '/res_{}_all.pdb'.format(lb_name)
        res_png_file = '/res_{}.png'.format(lb_name)

        RES = self.calc_results_for_labels(lbs,N)

        n_reported = RES["n_reported"]
        false_dets = RES["is_false"]
        true_dets =  RES["is_true"]
        prob_of_true = RES["prob_of_true"]
        probs_f = RES["probs_f"]
        dets_f = RES["dets_f"]
        xyz_f  = RES["xyz_f"]

        #True Pos vs False Pos
        plt.close()
        plt.subplot(2,1,1)
        plt.title('TRUE vs FALSE')
        h,= plt.plot(n_reported,prob_of_true,label = lb_name)
        plt.xlabel('Num of Reported')
        plt.ylabel('Probabillity of True')
        plt.legend(handles=[h],loc=4)
        plt.grid(True)
        plt.subplot(2,2,3)
        plt.title('RPOB vs TRUE and FALSE')
        h1,= plt.plot(1-probs_f,np.cumsum(false_dets),label = 'FALSE')
        h2,= plt.plot(1-probs_f,np.cumsum(true_dets),label = 'TRUE')
        plt.xlabel('1 - Probability')
        plt.ylabel('Num of Detections')
        plt.legend(handles=[h1,h2],loc=4)
        plt.grid(True)
        plt.subplot(2,2,4)
        plt.title('RPOB vs FA rate')
        h,= plt.plot(1-probs_f,np.cumsum(false_dets+0.01)/np.asarray(range(1,len(false_dets)+1)),label = lb_name)
        plt.xlabel('1 - Probability')
        plt.ylabel('Num of False Detections')
        plt.legend(handles=[h],loc=4)
        plt.grid(True)
        plt.subplots_adjust(top=0.9,bottom=0.1, left=0.10, right=0.95,hspace=0.35,wspace=0.35)
        plt.savefig(self.res_folder+res_png_file)
        plt.close()

        self.write_det_to_pdb_file(dets_f[:100],xyz_f[:100,:],filename=res_pdb_file)

        return


    def load_data(self):
        f = gzip.open(self.res_folder+self.file_name, 'rb')
        labels_names,xyz,results,centers_labels = cPickle.load(f)
        f.close()
        self.labels_names=labels_names
        self.xyz = xyz
        self.results = results
        self.centers_labels = centers_labels
        return

    def calc_referenced_labels(self,xyz,centers,labels):

        ref_labels =np.zeros(xyz.shape[0])
        dist_to_reference  = -1*np.ones(xyz.shape[0])
        if len(centers)==0:
            return ref_labels, dist_to_reference

        kdt = KDTree(centers)
        for inx in range(xyz.shape[0]):
            in_l = kdt.query_ball_point(xyz[inx,:],self.dist_thr)
            if len(in_l)>=1:
                ref_labels[inx] = labels[in_l[0]]
                dist_to_reference[inx] = np.linalg.norm(xyz[inx,:]-centers[in_l[0],:])

        return ref_labels, dist_to_reference



    def get_non_zero_preds(self):
        dets = np.argmax(self.results,axis=1)
        # selectet only non NONEs
        in_labels = np.where(dets!=0)[0]
        probs = np.asarray([self.results[in_labels[k]][dets[in_labels[k]]] for k in range(len(in_labels))])
        xyz=self.xyz[in_labels,:]
        dets = dets[in_labels]
        return dets,xyz,probs


    def create_over_all_text_res_file(self,filename = 'results.txt',N=1000):
        #calc detections
        dets,xyz,probs = self.get_non_zero_preds()
        #select and sort by Probability
        in_sort = np.argsort(probs)[::-1]
        in_sort = in_sort[:N]

        pred_labels = dets[in_sort]
        pred_conf =  probs[in_sort]
        pred_xyz =  xyz[in_sort,:]


        #filter detections
        probs_good,labels_good,xyz_good = self.filter_detections( pred_conf, pred_labels, pred_xyz,pred_conf, pred_labels, pred_xyz)

        ref_labels, dist_to_reference = self.calc_referenced_labels(xyz_good,self.centers_labels[0],self.centers_labels[1])

        f = open(self.res_folder+filename, 'w')
        f.write("PredictedLabel Confidence ReferenceLabel DistToRef X Y Z\n")
        for in_det in range(labels_good.shape[0]):
            f.write(self.labels_names[labels_good[in_det]])
            f.write(", ")
            f.write(str(probs_good[in_det]))
            f.write(", ")
            f.write(self.labels_names[ref_labels[in_det]])
            f.write(", ")
            f.write(str(dist_to_reference[in_det]))
            f.write(", ")
            f.write(str(xyz_good[in_det,0])+','+str(xyz_good[in_det,1])+','+str(xyz_good[in_det,2]))
            f.write(", ")
            f.write("\n")
        f.close()



        return

class DetRes4Nets(object):
    def __init__(self, res_objects,res_folder = TEMP_FOLDER):
        self.res_objects = res_objects
        self.res_folder = res_folder
        return
    def load_all_data(self):
        for res_obj in self.res_objects:
            res_obj.load_data()

    def save_all_data(self):
        for res_obj in self.res_objects:
            res_obj.save_data()

    def plot_results_per_label(self,lbs,N=1000,x_max = 50):

        #True Pos vs False Pos
        lb_name = [self.res_objects[0].labels_names[lb] + '_' for lb in lbs]
        res_png_file = '/res_{}.png'.format(lb_name)
        plt.title('Detection Ratio: ' + str(lb_name))
        hndls=[]
        plt.close()
        for ro in self.res_objects:
            RES = ro.calc_results_for_labels(lbs,N)
            n_reported = RES["n_reported"]
            false_dets = RES["is_false"]
            true_dets =  RES["is_true"]
            prob_of_true = RES["prob_of_true"]
            h,= plt.plot(n_reported,prob_of_true,label = ro.name)
            hndls.append(h)

        plt.xlabel('Num of Reported')
        plt.ylabel('Probabillity of True')
        plt.legend(handles=hndls,loc=4)
        plt.grid(True)
        plt.xlim(0, x_max)
        plt.savefig(self.res_folder+res_png_file)
        plt.close()
        return




class SingleNetResults(object):
    def __init__(self, res_folder, labels_names,res_per_epoch=[[]], valid_data=[[],[]], prob_span =  np.arange(0.05,1.0001,0.1), train_data_stat = [],name = 'Untitled'):
        self.res_per_epoch=res_per_epoch
        self.res_folder = res_folder
        self.valid_data = valid_data
        self.train_data_stat = train_data_stat
        self.det_fa_per_epoch = []
        self.labels_names = labels_names
        self.dets_per_epoch = []
        self.det_acc_per_epoch = []
        self.det_fa_per_epoch = []
        self.prob_span = prob_span
        self.labels = labels_names.keys()
        self.names = labels_names.values()
        self.y = []
        self.dets_per_epoch= []
        self.prob_vs_output = {}
        self.det_matrix = []
        self.name = name
        return

    def save_data(self,file_name = 'res_per_epoch.pkl.gz'):
        f = gzip.open(self.res_folder+file_name, "w")
        cPickle.dump((self.res_per_epoch,self.valid_data,self.train_data_stat), f)
        f.close()
        return

    def load_data(self,file_name = 'res_per_epoch.pkl.gz'):
        f = gzip.open(self.res_folder+file_name, 'rb')
        res_per_epoch, valid_data,train_data_stat = cPickle.load(f)
        f.close()
        self.train_data_stat = train_data_stat
        self.res_per_epoch = res_per_epoch
        self.valid_data = valid_data
        self.calc_results()
        return

    def calc_results(self):
        self.labels = self.labels_names.keys()
        self.names = self.labels_names.values()
        self.y = np.asarray(self.valid_data[1])
        self.dets_per_epoch= [self.calc_detection(x) for x in self.res_per_epoch]
        self.det_acc_per_epoch = [self.calc_detection_accuracy(x, self.y,self.labels ) for x in self.dets_per_epoch]
        self.det_fa_per_epoch = [self.calc_FA_rate(x, self.y,self.labels ) for x in self.dets_per_epoch]

        self.prob_vs_output = {}
        for lb in self.labels:
            prob_x, det_y,fa_y = self.calc_det_vs_prob(lb,self.res_per_epoch[-1][:,lb],self.y,self.prob_span )
            self.prob_vs_output[lb]={'prob_x':prob_x,'det_y':det_y,'fa_y':fa_y}


        self.det_matrix = self.calc_detection_matrix(self.dets_per_epoch[-1],
                                                    self.y,self.labels)
        return

    def calc_detection(self,soft_max_probs):
        det = np.argmax(soft_max_probs,axis=1)
        return det

    def calc_detection_accuracy(self,dets, y,labels ):
        N = np.array([np.sum(y == lb) for lb in labels ],dtype=float) +0.0001
        true_detections = dets == y
        n_det=np.array([np.sum((true_detections) & (y == lb)) for lb in labels])
        det_acc = []
        return n_det/N

    def calc_det_vs_prob(self,lb,probs,y,prob_span = np.arange(0,1+0.1,0.1)):

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

    def calc_detection_matrix(self,dets,y,lbs):
        det_mat = np.zeros((len(lbs),len(lbs)),dtype=float)
        for lb_true in lbs:
            in_true = y == lb_true
            n_true = np.sum(in_true)+0.001
            for lb_det in lbs:
                in_det = dets==lb_det
                det_mat[lb_true,lb_det] = np.sum(in_det & in_true)/n_true
        return det_mat




    def calc_FA_rate(self,dets, y,labels):
        """
        False Alarm Rate = m/n
        m - number of false positive detections
        n - number  of samples of other labels
        """
        N = np.array([np.sum(y != lb) for lb in labels ],dtype=float)+0.0001
        falses_detections = dets != y
        n_fa=np.array([np.sum((falses_detections) & (y == lb)) for lb in labels])
        return n_fa/N

    def save_detection_martix_plot(self,filename = 'det_mtrx.png'):
        plt.close()
        plt.imshow(self.det_matrix,interpolation = 'none',cmap = 'nipy_spectral')
        plt.clim(0,1.0)
        plt.colorbar()
        plt.xticks(self.labels,self.names,rotation=90)
        plt.yticks(self.labels,self.names)
        plt.title('column I, column J = Prob(Detecting AA of type I, given AA of type J ]')
        ax = plt.gca()
        ax.set_xticks(np.array(self.labels)+0.5, minor=True)
        ax.set_yticks(np.array(self.labels)+0.5, minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        plt.savefig(self.res_folder+filename)

    def plot_det_vs_datasize(self):
        plt.close()
        file_name = 'acc_vs_size.png'
        det_acc = self.det_acc_per_epoch[-1]
        train_data_size = [self.train_data_stat.get(x,0) for x in range(len(self.labels))]
        plt.plot(train_data_size,det_acc,marker = 's',linestyle = 'None')
        for x in range(len(self.labels)):
            plt.text(train_data_size[x], det_acc[x], self.labels_names[x], fontsize=8)
        plt.title('Detection Accuracy vs Train Data Size')
        plt.savefig(self.res_folder+file_name)

    def save_detection_graphs_one_run(self):
        self.save_detection_martix_plot()

        epoch_x_axis = range(len(self.det_acc_per_epoch))
        for lb in self.labels:
            file_name = '/res_'+self.labels_names[lb] +'.png'
            det_rate_one_label = [x[lb]  for x in self.det_acc_per_epoch]
            fa_rate_one_label = [x[lb]  for x in self.det_fa_per_epoch]

            plt.subplot(2,1,1)
            plt.title('Detection Accuracy' + self.labels_names[lb])
            det_ln,=plt.plot(epoch_x_axis,det_rate_one_label,label='detection rate')
            fa_ln,=plt.plot(epoch_x_axis,fa_rate_one_label,label='FA rate')
            plt.xlabel('epoch')
            plt.legend(handles=[det_ln, fa_ln],loc=3)
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.title('Last Epoch SoftMax Output (Probability) ' + self.labels_names[lb])
            det_ln,=plt.plot(self.prob_vs_output[lb]['prob_x'] ,self.prob_vs_output[lb]['det_y'],label='Prob of True')
            fa_ln,=plt.plot(self.prob_vs_output[lb]['prob_x'] ,self.prob_vs_output[lb]['fa_y'],label='Prob of False')
            plt.xlabel('Probability')
            plt.legend(handles=[det_ln, fa_ln],loc=3)
            plt.grid(True)
            plt.savefig(self.res_folder+file_name)
            plt.close()

        self.plot_det_vs_datasize()

# class ClassResults4Net(object):
#
#     def __init__(self, res_folder, res_nets,prob_span =  np.arange(0.05,1.0001,0.025)):
#         super(Employee, self).__init__
#
#     def calc_results():
#         for rs in rs_nets:
#             rs.calc_results()
#         return
#
#     def save_data():
#         return
#
#     def plot_detection_matrix:
#         plt.close()
#
#         for in range(4)
#         plt.imshow(self.det_matrix,interpolation = 'none',cmap = 'nipy_spectral')
#         plt.clim(0,1.0)
#         plt.colorbar()
#         plt.xticks(self.labels,self.names,rotation=90)
#         plt.yticks(self.labels,self.names)
#         plt.title('column I, column J = Prob(Detecting AA of type I, given AA of type J ]')
#         ax = plt.gca()
#         ax.set_xticks(np.array(self.labels)+0.5, minor=True)
#         ax.set_yticks(np.array(self.labels)+0.5, minor=True)
#         ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
#         plt.savefig(self.res_folder+filename)
