from chimera import runCommand
import chimera
from MarkChimeraUtils import atomsList2spec
import VolumeViewer
import glob
import numpy as np
import os, sys
import time

#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path +'/../python/'
sys.path.append(python_path)
import dataset_loader
reload(dataset_loader)
from dataset_loader import read_list_file, get_file_names,VX_FILE_SUFF
from dataset_loader import VOX_SIZE, RESOLUTION,NBOX_IN, N_SAMPLS_FOR_1V3




def read_list_file_old(list_file):
    emdcodes=[]
    pdbcodes =[]
    with open(list_file) as fp:
        for cnt, line in enumerate(fp):
            words = line.split()
            emdcodes.append(words[0])
            pdbcodes.append(words[-1])

    return zip((emdcodes, pdbcodes))

def get_prot_gabarites(pdb_id):

    prot = get_object_by_id(pdb_id)

    (x_min,x_max,y_min,y_max,z_min,z_max)=(10.0**6,-10.0**6,10.0**6,-10.0**6,10.0**6,-10.0**6)
    for at in prot.atoms:
        x_min = min(at.coord()[0],x_min)
        x_max = max(at.coord()[0],x_max)
        y_min = min(at.coord()[1],y_min)
        y_max = max(at.coord()[1],y_max)
        z_min = min(at.coord()[2],z_min)
        z_max = max(at.coord()[2],z_max)

    return x_min,x_max,y_min,y_max,z_min,z_max


def calc_random_sample(lcc_mtrx, pdb_id):
    x_min,x_max,y_min,y_max,z_min,z_max = get_prot_gabarites(pdb_id)

    N_reqiured = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)*N_SAMPLS_FOR_1V3
    N_found = np.count_nonzero(lcc_mtrx>0)

    del_ratio = (N_found-N_reqiured)/N_found
    del_ratio = max(0,del_ratio)

    lcc_mtrx = np.greater(lcc_mtrx,0.0)*1.0

    rnd_sampl = np.random.random(lcc_mtrx.shape)-del_ratio;
    lcc_mtrx = lcc_mtrx*rnd_sampl
    lcc_mtrx = np.greater(lcc_mtrx,0.0)*1.0

    return lcc_mtrx

def calc_LCC(pdb_id,map_id,grid3D):
    sim_map_id = 7001
    res_for_simulation=3
    lcc_map_id = 7002
    #simulate map
    runCommand('molmap #{} {}  modelId {}'.format(pdb_id,res_for_simulation,sim_map_id))
    #calc_LCC
    runCommand('vop localCorrelation  #{} #{} windowSize {} subtractMean true  modelId {} '.\
                       format(sim_map_id,map_id,NBOX_IN,lcc_map_id))
    lcc_mtrx = map_to_matrix(lcc_map_id,grid3D)
    #get points above mean
    lcc_mtrx[:(NBOX_IN+1)/2,:,:]=0.0
    lcc_mtrx[-(NBOX_IN+1)/2:,:,:]=0.0
    lcc_mtrx[:,:(NBOX_IN+1)/2,:]=0.0
    lcc_mtrx[:,-(NBOX_IN+1)/2:,:]=0.0
    lcc_mtrx[:,:,:(NBOX_IN+1)/2]=0.0
    lcc_mtrx[:,:,-(NBOX_IN+1)/2:]=0.0
    lcc_mtrx = lcc_mtrx-np.mean(lcc_mtrx)

    #
    lcc_mtrx = lcc_mtrx-np.mean(lcc_mtrx)
    lcc_mtrx = np.greater(lcc_mtrx,0.0)*1.0
    return lcc_mtrx


def calc_voxalization_by_atom_type(pdb_id,grid3D,res=RESOLUTION):

    Id_for_copy = 4001
    Id_for_molmap = 5001

    vx_map = {}

    ##loop on atoms
    for at_name in VX_FILE_SUFF.keys():

        ##copy structure
        runCommand('combine #{} name  {}_atoms modelId {}'.\
                   format(pdb_id, at_name,Id_for_copy))
        ## delete atoms
        runCommand('delete #{}:@/element!={}'.format(Id_for_copy,at_name))
        ## run molmap
        no_atoms = get_object_by_id(Id_for_copy)==-1


        if no_atoms:
            runCommand('vop new zero_map origin {},{},{} modelId {}'.\
                       format(np.mean(grid3D[0]),np.mean(grid3D[1]),np.mean(grid3D[1]),Id_for_molmap))
        else:
            runCommand('molmap #{} {}  modelId {}'.\
                   format(Id_for_copy,res,Id_for_molmap))

        #extract matrix (copy?)
        vx_map[at_name]=map_to_matrix(Id_for_molmap,grid3D)

        # delete copied structure
        runCommand('close #{}'.format(Id_for_copy))
        # delete mol map
        runCommand('close #{}'.format(Id_for_molmap))

    return vx_map

def get_object_by_id(id):
    all_objs = filter(lambda x:x.id==id, chimera.openModels.list())
    if len(all_objs)==1:
        return all_objs[0]
    else:
        return -1

def calc_3D_grid(pdb_id,vx_size,res):

    syth_map_id = 6001
    #molmap
    runCommand('molmap #{} {} gridSpacing {} modelId {} replace false '\
               .format(pdb_id,res,vx_size,syth_map_id))
    #extract grid
    v_obj = get_object_by_id(syth_map_id)

    Xmin,Xmax = v_obj.xyz_bounds()
    xr = np.arange(Xmin[0]+vx_size/3,Xmax[0],vx_size)
    yr = np.arange(Xmin[1]+vx_size/3,Xmax[1],vx_size)
    zr = np.arange(Xmin[2]+vx_size/3,Xmax[2],vx_size)
    Xs,Ys,Zs = np.meshgrid(xr,yr,zr,indexing='ij')

    #remove models
    runCommand('close #{}'.format(syth_map_id))
    #return output
    return Xs,Ys,Zs

def map_to_matrix(map_id,grid3D):
    v_obj = get_object_by_id(map_id)
    Xs,Ys,Zs = grid3D
    xyz_coor = np.vstack((np.reshape(Xs,-1),np.reshape(Ys,-1),np.reshape(Zs,-1))).transpose()
    values, outside = v_obj.interpolated_values(xyz_coor,out_of_bounds_list = True)
    mtrx = np.reshape(values,Xs.shape)

    return mtrx

def calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION):
    prot1 = chimera.openModels.open(pdb_file)[0]
    map_obj = VolumeViewer.volume.open_volume_file(map_file)[0]
    pdb_id = prot1.id
    map_id = map_obj.id

    #add hydrogens
    runCommand('addh spec #{}'.format(pdb_id))

    Xs,Ys,Zs = calc_3D_grid(pdb_id,vx_size,res)

    output_mtrx = map_to_matrix(map_id,(Xs,Ys,Zs))
    lcc_mtrx = calc_LCC(pdb_id,map_id,(Xs,Ys,Zs))
    lcc_mtrx = calc_random_sample(lcc_mtrx, pdb_id)
    inp_mtrc = calc_voxalization_by_atom_type(pdb_id,(Xs,Ys,Zs),res=res)

    return inp_mtrc, output_mtrx, lcc_mtrx

def save_matrc_to_folder(folder_name,pdb_id,inp_mtrc,output_mtrx,lccc_mtrx):
    f_names = get_file_names(pdb_id,folder_name)
    np.save(f_names['OUT'],output_mtrx)
    np.save(f_names['LCC'],lccc_mtrx)

    for at_name in VX_FILE_SUFF.keys():
        np.save(f_names[at_name],inp_mtrc[at_name])
    return

def create_database(input_folder, output_folder, list_file):
    # read list file
    pairs = read_list_file(list_file)
    for pair in pairs:
        runCommand('close all')
        map_id = pair[1]
        pdb_id = pair[0]
        # get files
        map_file = glob.glob(input_folder+'/emd-{}.*'.format(map_id))[0]
        pdb_file = glob.glob(input_folder+'/{}.*'.format(pdb_id))[0]
        #calc voxalization
        inp_mtrc, output_mtrx, local_fit_matrix = calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION)
        #save data to folder
        save_matrc_to_folder(output_folder,pdb_id,inp_mtrc, output_mtrx, local_fit_matrix)





if __name__=='main':
    runCommand('close all')
    #pdb_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/6j2c.cif"
    #map_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/emd-2984.map"
    pdb_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/test1.pdb"
    map_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/t1.mrc"
    #
    list_file = '/Users/markroza/Documents/work_from_home/NNcourse_project/data/res6/synth/list.txt'
    input_folder = '/Users/markroza/Documents/work_from_home/NNcourse_project/data/res6/synth/'
    output_folder = input_folder
    pp = read_list_file(list_file)
    create_database(input_folder, output_folder, list_file)
