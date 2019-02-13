import os
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


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
ch_path = dir_path + '/../../chimeracode/'
sys.path.append(utils_path)
sys.path.append(ch_path)

from dbcreator import DBcreator,EMmaps
from dbloader import Mean0Sig1Normalization, NoNormalization,load_det_labeled_5tuple_data


def box_from_box_center(x,y,z,cbs=11,apx=1):

    x_gr = np.linspace(x-cbs/2*apx,x+cbs/2*apx,cbs)
    y_gr = np.linspace(y-cbs/2*apx,y+cbs/2*apx,cbs)
    z_gr = np.linspace(z-cbs/2*apx,z+cbs/2*apx,cbs)
    X,Y,Z = np.meshgrid(x_gr,y_gr,z_gr,indexing = 'ij')
    box_xyz = (X.astype(int),Y.astype(int),Z.astype(int))

    return box_xyz



runCommand('close all')

rand_pos = 1.0
sfx = "_rand1.pkl.gz"

data_folder = dir_path + '/../../data/'
input_folder = data_folder+'/cryoEM/raw_data/res0023_aug10/'
target_folder = data_folder+'/temp/'

#dbc = DBcreator(input_pdb_folder = input_folder, mrc_maps_folder = input_folder,target_folder = target_folder,normalization = Mean0Sig1Normalization,rand_pos= rand_pos)

dbc = DBcreator(input_pdb_folder = input_folder, mrc_maps_folder = input_folder,target_folder = target_folder,normalization = NoNormalization,rand_pos= rand_pos,cubic_box_size=3)

pos_reference = [100,100,100]

pos = [67,105,110]
bx_ref = box_from_box_center(pos_reference[0],pos_reference[1],pos_reference[2],cbs=3)
bx_cnt = (bx_ref[0]-pos_reference[0]+pos[0],bx_ref[1]-pos_reference[1]+pos[1],bx_ref[2]-pos_reference[2]+pos[2])

box1 = EMmaps.extract_3D_boxes(input_folder+"emd-2984.map",[bx_cnt],NoNormalization)
print "BOX CENTER"
print bx_cnt

print "BOX1"
print box1


DATA_FILE = '/specific/netapp5_2/iscb/wolfson/Mark/Projects/NNcryoEM/data/temp/det2984.pkl.gz'
data_mtrx,filter_matrix,C,centers,labels = load_det_labeled_5tuple_data(DATA_FILE)


#data_reorder = np.zeros(data_mtrx.shape[3],data_mtrx.shape[2],data_mtrx.shape[1])
#for i in range(data_mtrx.shape[0]):
#    for j in range(data_mtrx.shape[1]):
#        for k in range(data_mtrx.shape[2]):
#            box2[i,j,k]=data_mtrx[bx_cnt[2][i,j,k],bx_cnt[1][i,j,k],bx_cnt[0][i,j,k]]

box2 = np.zeros((3,3,3))
for i in range(box2.shape[0]):
    for j in range(box2.shape[1]):
        for k in range(box2.shape[2]):
            box2[i,j,k]=data_mtrx[bx_cnt[2][i,j,k],bx_cnt[1][i,j,k],bx_cnt[0][i,j,k]]
print "BOX2"
print box2

print "DIFF"
print box2-box1

box3 = data_mtrx[bx_cnt[2].astype(int),bx_cnt[1].astype(int),bx_cnt[0].astype(int)]



print "BOX3"

print "DIFF2"
print box2-box1
print box3-box2

data_reorder = np.swapaxes(data_mtrx,0,2)
box4 = data_reorder[bx_cnt[0],bx_cnt[1],bx_cnt[2]]
print "DIFF3"
print box2-box1
print box3-box2
print box4-box1

#dbc.create_classification_db("emd-2984.map","5a1a.pdb",file_name_suffix = sfx )
