import os
import shutil
import sys
import time
import numpy as np
import random
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import open_volume_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cur_pass = os.path.realpath(__file__)

utils_path = cur_pass + '/../pythoncode/utils/'
chimera_path = cur_pass + '/../chimeracode/'
sys.path.append(utils_path)


import kdtree
reload(kdtree)
from kdtree import KDTree_3D_objects
from MarkChimeraUtils import atomsList2spec
THR_DIST = 2.0

## Mark True /False
def calcCg_of_res(residue):
    cg =np.array([0,0,0])
    natoms = 0.0
    for atom in residue.atoms:
        cg = cg + atom.coord()
        natoms+=1
    cg = cg/natoms
    return cg[0],cg[1],cg[2]




def load_data(emd_map_file, res_prot_file, ref_pdb_file = None):

    if ref_pdb_file != None:
        ref_prot = chimera.openModels.open(ref_pdb_file,'PDB')
    else:
        ref_prot = None
    res_prot = chimera.openModels.open(res_pdb_file,'PDB')[0]
    whole_map = open_volume_file(emd_map_file,model_id=17)[0]
    return whole_map, res_prot, ref_prot

def color_map(map, res_prot):
    runCommand("vop zone #{}  #{}  3".format(map.id,res_prot.id))



def test_results(out_pdb_file, ref_pdb_file , out_file = 't1.txt', is_plot=True):

    #load files
    ref_prot = chimera.openModels.open(ref_pdb_file,'PDB')
    out_prot = chimera.openModels.open(out_pdb_file,'PDB')[0]
    # create -kdtree
    res_data=[]
    res_cg = []

    for ref_model in ref_prot:
        for res in ref_model.residues:
            res_data.append(res)
            res_cg.append(calcCg_of_res(res))

    kdt4 = KDTree_3D_objects(res_cg,res_data)

    #get all res types
    res_types = list(set([x.type for x in out_prot.residues]))
    res_data = {}

    with open(out_file, 'w') as f:
        for residue_type in res_types:
            out_residues = filter(lambda x: x.type==residue_type, out_prot.residues )
            res_data[residue_type] = {'TF':[], 'PseudoProb':[]}
            print(residue_type)
            n_true=0
            n_false=0
            f.write(residue_type+'\n')
            for res in out_residues:
                atm = res.findAtom('K')
                position = atm.coord()
                f.write(str(atm.residue.id.position) + ' : ' +str(atm.bfactor)+' ### ')
                res_data[residue_type]['PseudoProb'].append(atm.bfactor)
                inxs, ress = kdt4.in_range(position,THR_DIST)
                if len(inxs) !=1:
                    f.write( '  FALSE Num Near{}'.format(len(inxs)))
                    n_false+=1
                    res_data[residue_type]['TF'].append(0)
                else:
                    res_found = ress[0]
                    if res_found.type != residue_type:
                        n_false+=1
                        f.write( '  FALSE UNCORRECT TYPE :{}'.format(res_found.type))
                        res_data[residue_type]['TF'].append(0)
                    else:
                        f.write( '  TRUE ' +res_found.id.chainId + ' ' +str(res_found.id.position) )
                        n_true+=1
                        res_data[residue_type]['TF'].append(1)

                f.write('  ' + str(position)+'\n')
            f.write('Summary Confidence  ' + str(n_true/(n_true+n_false+0.0001))+'\n')
            print('Conf', n_true/(n_true+n_false+0.0001))

    if is_plot:
        plot_results_per_label(res_data,out_file)


def plot_results_per_label(res_data, res_file):
    #get full path
    folder_path = os.path.dirname(res_file)
    graph_path = folder_path+'/graphs/'
    #remove old res_folder
    print("DEBUG 100")
    shutil.rmtree(graph_path, ignore_errors=True)
    print("DEBUG 1200")

    #create new folder
    os.mkdir(graph_path)


    for res_type in res_data.keys():
        probs = np.array(res_data[res_type]['PseudoProb'])
        tf = np.array(res_data[res_type]['TF'])

        rand = np.random.random(tf.shape)+10.0
        in_sorted = np.argsort(probs)
        probs_sorted = probs[in_sorted]
        tf_sorted = tf[in_sorted]
        n_det= np.array(range(len(in_sorted))) + 1
        n_true_asc = np.cumsum(tf_sorted)
        n_true = n_true_asc[-1] - n_true_asc
        n_all = n_det[-1]-n_det+1
        conf = (n_true+0.0)/n_all

        #True Pos vs False Pos
        plt.close()
        plt.subplot(2,1,1)
        plt.title('Confidence vs Prob')
        h,= plt.plot(probs_sorted,conf,label = 'Confidence')
        plt.xlabel('Threshold for Report')
        plt.ylabel('Probabillity of True')
        #plt.legend(handles=[h],loc=4)
        plt.subplot(2,1,2)
        plt.title('Num detections vs Prob')
        h,= plt.plot(probs_sorted,n_all,label = 'Number of detections')
        plt.xlabel('Threshold for Report')
        plt.ylabel('Number of detections')
        #plt.legend(handles=[h],loc=4)
        plt.grid(True)
        # plt.subplot(2,2,3)
        # plt.title('RPOB vs TRUE and FALSE')
        # h1,= plt.plot(1-probs_f,np.cumsum(false_dets),label = 'FALSE')
        # h2,= plt.plot(1-probs_f,np.cumsum(true_dets),label = 'TRUE')
        # plt.xlabel('1 - Probability')
        # plt.ylabel('Num of Detections')
        # plt.legend(handles=[h1,h2],loc=4)
        # plt.grid(True)
        # plt.subplot(2,2,4)
        # plt.title('RPOB vs FA rate')
        # h,= plt.plot(1-probs_f,np.cumsum(false_dets+0.01)/np.asarray(range(1,len(false_dets)+1)),label = lb_name)
        # plt.xlabel('1 - Probability')
        # plt.ylabel('Num of False Detections')
        # plt.legend(handles=[h],loc=4)
        # plt.grid(True)
        # plt.subplots_adjust(top=0.9,bottom=0.1, left=0.10, right=0.95,hspace=0.35,wspace=0.35)
        plt.savefig(graph_path +'/' + res_type+'.png')
        plt.close()

    return



def test_one_label(out_pdb_file, ref_pdb_file , res_type,thr):
    #load files
    ref_prot = chimera.openModels.open(ref_pdb_file,'PDB')
    out_prot = chimera.openModels.open(out_pdb_file,'PDB')[0]
    # create -kdtree
    res_data=[]
    res_cg = []

    for ref_model in ref_prot:
        for res in ref_model.residues:
            res_data.append(res)
            res_cg.append(calcCg_of_res(res))

    kdt4 = KDTree_3D_objects(res_cg,res_data)

    #get all res types
    out_residues = filter(lambda x: x.type==res_type, out_prot.residues )
    print(res_type)
    n_true=0
    n_false=0
    for res in out_residues:
        atm = res.findAtom('K')
        if atm.bfactor<thr:
            continue
        position = atm.coord()
        inxs, ress = kdt4.in_range(position,THR_DIST)
        if len(inxs) !=1:
            n_false+=1
        else:
            res_found = ress[0]
            if res_found.type != res_type:
                n_false+=1
            else:
                n_true+=1
    print('Conf', (n_true+0.0001)/(n_true+n_false+0.0001))


def main():
    ref_pdb_file = '/Users/markroza/Documents/work_from_home/aanchor_server/input_files/res22/pdb5a1a.ent.txt'
    out_pdb_file = '/Users/markroza/Documents/work_from_home/aanchor_server/upload/emd2984gzpklY2019M02D11H10MMS50/results.pdb'
    emd_file = '/Users/markroza/Documents/work_from_home/aanchor_server/input_files/res22/emd-2984.map'

    test_results(out_pdb_file, ref_pdb_file , out_file = '/Users/markroza/Documents/work_from_home/aanchor_server/temp/t1.txt')

    error


    runCommand('close all')
    THR_DIST = 1.5

    res_type = 'LEU'
    res_color = 'red'

    em_map, res_prot, ref_prot = load_data(emd_file, res_pdb_file)

    #delete unwanted types
    atoms_to_delete = filter(lambda x: x.residue.type != res_type, res_prot.atoms)
    runCommand('delete {}'.format(atomsList2spec(atoms_to_delete)))

    #color map
    color_map(em_map,res_prot)







if __name__ == "chimeraOpenSandbox":
    n_command = sys.argv[3]
    if n_command == 'test':
        out_file = sys.argv[4]
        ref_file = sys.argv[5]
        results_file = sys.argv[6]
        test_results(out_file, ref_file , out_file = results_file, is_plot = False)

    if n_command == 'test_graphs':
        out_file = sys.argv[4]
        ref_file = sys.argv[5]
        results_file = sys.argv[6]
        test_results(out_file, ref_file , out_file = results_file, is_plot = True)

    if n_command == 'test_one':

        out_file = sys.argv[4]
        ref_file = sys.argv[5]
        res_type = sys.argv[6]
        res_thr = float(sys.argv[7])

        test_one_label(out_file, ref_file , res_type, res_thr)
    runCommand('stop')








## Statistcs by label

## Show
