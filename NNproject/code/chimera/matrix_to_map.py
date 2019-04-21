from chimera import runCommand
import chimera
import VolumeViewer
import glob
import numpy as np
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
import os, sys

#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path +'/../python/'
sys.path.append(python_path)
sys.path.append(dir_path)



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
    step = [vx_size,vx_size,vx_size]
    mtrx_1 = np.swapaxes(mtrx,2,0)

    grid = Array_Grid_Data(mtrx_1, org, step, name = name)
    v = volume_from_grid_data(grid)

    return v

def save_map(mp, folder):
        file_name = folder + mp.name + '.mrc'
        runCommand('volume #{}  save {}'.format(mp.id,file_name))

def matrix_to_map(input_file,vx_size,map_name,out_folder):
    mtrx = np.load(input_file)
    v = create_map(mtrx, vx_size, org = [0.0,0.0,0.0], name = map_name)
    save_map(v, out_folder)


if __name__ == "chimeraOpenSandbox":
    k=0
    while not __file__[0:10] in sys.argv[k]:
        k=k+1
        print sys.argv[k], '###', __file__
    command = sys.argv[k+1]
    vx_size = float(sys.argv[k+2])
    input_file = sys.argv[k+3]
    map_name = sys.argv[k+4]
    out_folder = sys.argv[k+5]

    if command == "mtrx2map":
        matrix_to_map(input_file,vx_size,map_name,out_folder)

    runCommand('stop')
