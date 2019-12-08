from chimera import runCommand
import chimera
import VolumeViewer
import glob
import numpy as np
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
import os, sys

python_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/python/'
chimera_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera/'
sys.path.append(python_path)
sys.path.append(chimera_path)

from dataset_loader import VOX_SIZE


def create_map(mtrx, vx_size, org = [0.0,0.0,0.0], name = 'Marik'):
    step = [vx_size,vx_size,vx_size]
    mtrx_1 = np.swapaxes(mtrx,2,0)

    mtrx_1[np.where(mtrx_1==0)]=-1

    grid = Array_Grid_Data(mtrx_1, org, step, name = name)
    v = volume_from_grid_data(grid)

    return v

def save_map(mp, folder):
        file_name = folder + mp.name + '.mrc'
        runCommand('volume #{}  save {}'.format(mp.id,file_name))

def matrix_to_map(input_file,out_file):

    head, tail = os.path.split(out_file)

    map_name = tail[:-4]
    out_folder = head+'/'

    mtrx = np.load(input_file)
    v = create_map(mtrx, VOX_SIZE, org = [0.0,0.0,0.0], name = map_name)
    save_map(v, out_folder)


if __name__ == "chimeraOpenSandbox":
    k=0
    while not __file__[0:10] in sys.argv[k]:
        k=k+1
        print sys.argv[k], '###', __file__

    input_npy_file = sys.argv[k+1]
    out_mrc_file = sys.argv[k+2]

    matrix_to_map(input_npy_file,out_mrc_file)

    runCommand('stop')
