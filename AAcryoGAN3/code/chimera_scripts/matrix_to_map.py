from chimera import runCommand
import chimera
import VolumeViewer
import glob
import numpy as np
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
import os, sys
from VolumeViewer import open_volume_file
import time


#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path +'/../python/'
chimera_path = dir_path +'/../chimera/'

sys.path.append(python_path)
sys.path.append(chimera_path)

import utils_project
import dataset_loader


def old_implimentation():

    runCommand('close all')
    import createDB
    reload (createDB)
    import dataset_loader
    reload(dataset_loader)
    from dataset_loader import read_list_file, get_file_names,VX_FILE_SUFF
    from dataset_loader import VOX_SIZE, RESOLUTION,NBOX_IN, N_SAMPLS_FOR_1V3


    #load map
    pdb_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/single_pdbs/alll.pdb"
    test_map_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/temp_data/test1.npy"
    #init_map = '/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/single_pdbs/emd-1111.mrc'
    mtrx_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/single_pdbs/F_alll_output.npy"

    #load pdb
    prot1 = chimera.openModels.open(pdb_file)[0]
    #create map
    res = 1.5
    vx_size = 1.5
    runCommand('molmap #{} {} gridSpacing {} modelId {}'.format(prot1.id,res,vx_size,1001))

    m1 = createDB.get_object_by_id(1001)
    org, step = m1.data_origin_and_step()

    Xs,Ys,Zs = createDB.calc_3D_grid(prot1.id,vx_size,res)
    mtrx_in = createDB.map_to_matrix(1001,(Xs,Ys,Zs))
    #save map
    np.save(test_map_file,mtrx_in)


    mtrx = np.load(test_map_file)


def create_map(mtrx, vx_size, org = [0.0,0.0,0.0], name = 'Marik'):
    runCommand('close all')

    step = [vx_size,vx_size,vx_size]
    mtrx_1 = np.swapaxes(mtrx,2,0)

    mtrx_1[np.where(mtrx_1==0)]=-1

    grid = Array_Grid_Data(mtrx_1, org, step, name = name)
    v = volume_from_grid_data(grid)

    return v

def fit_pdb(old_pdb_file, mrc_file, new_pdb_file):

    prot = chimera.openModels.open(old_pdb_file)[0]
    gan_map = open_volume_file(mrc_file,model_id=17)[0]
    #map center
    map_cent = gan_map.data.ijk_to_xyz(np.array(gan_map.data.mrc_data.matrix_size)/2)
    #pdb center
    cg = np.array([0,0,0])
    nn = 0
    for atom in prot.atoms:
        cg = cg+atom.coord()
        nn=nn+1
    cg = cg/nn
    # move pdb to map
    df = map_cent-cg
    for atom in prot.atoms:
        crd = atom.coord()
        atom.setCoord(chimera.Point(crd[0]+df[0],crd[1]+df[1],crd[2]+df[2]))
    # run fit to map
    for i in range(20):
        runCommand('fitmap #{} #17 rotate false '.format(prot.id))
        time.sleep(2)
    #save pdb
    runCommand('write relative #17 format pdb {} {}'.format(prot.id, new_pdb_file))

    runCommand('close all')
    return

def matrix_to_map(input_file, out_file,vx_size):
    mtrx = np.load(input_file)

    v = create_map(mtrx, vx_size, org = [0.0,0.0,0.0], name = "gan")
    runCommand('volume #{}  save {}'.format(v.id,out_file))



if __name__ == "chimeraOpenSandbox":
    k=0
    while os.path.basename(sys.argv[k])[0:5]!= __file__[0:5]:
        k=k+1
        print sys.argv[k], '###', __file__
    inp_list_file   = sys.argv[k+1]
    out_list_file = sys.argv[k+2]
    npy_folder   = sys.argv[k+3]
    mrc_folder  = sys.argv[k+4]
    pdb_folder  = sys.argv[k+5]
    resolution  = np.float(sys.argv[k+6])
    vx_size     = np.float(sys.argv[k+7])

    print("input list_file:",inp_list_file)
    print("output list_file:",out_list_file)
    print("npy_folder:",npy_folder)
    print("pdb_file:", pdb_folder)
    print("mrc_folder:",mrc_folder)
    print("resolution:",resolution)
    print("vx_size:",vx_size)


    utils_project.assert_vx_size_and_resolution(vx_size,resolution)

    with open(out_list_file,"w") as f:
        f.write("PDB_ID MAP_ID RESOLUTION TRAIN/TEST/VALID/ IS_VIRUS REMARK\n")


    pairs = utils_project.read_list_file(inp_list_file)
    for pair in pairs:
        pdb_file = pair["pdb_file"]
        emd_file = pair["emd_file"]
        out_file = mrc_folder + '/' + emd_file
        npy_file_name = dataset_loader.get_file_names(pdb_file,npy_folder)['GAN_NPY']
        mrc_file_name = dataset_loader.get_file_names(emd_file,mrc_folder)["MRC"]
        f_pdb_fit     = dataset_loader.get_file_names(pdb_file,mrc_folder)["PDB_FIT"]

        matrix_to_map(npy_file_name, mrc_file_name, vx_size)
        old_pdb_file = pdb_folder +'/'+pdb_file
        fit_pdb(old_pdb_file, mrc_file_name, f_pdb_fit)

        with open(out_list_file,"a") as f:
            gan_mrc_file = os.path.basename(mrc_file_name)
            new_pdb_file = os.path.basename(f_pdb_fit)
            remark = "GAN CREATED"
            f.write(new_pdb_file+" " + gan_mrc_file+" " +str(pair["res"])+" "+pair["train_test"]+\
            " "+pair["is_virus"]+" "+remark+"\n")


    runCommand('stop')
