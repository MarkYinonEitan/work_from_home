import chimera
from chimera import specifier
import VolumeViewer
from Matrix import euler_xform
from chimera import runCommand
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '../../../code/chimera/'
utils_path= '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera/'
sys.path.append(utils_path)

import EM_utils
reload(EM_utils)
from EM_utils import get_box, calc_box_grid, cut_box_gx


data_folder = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/'
res_folder = data_folder+'look_for_prove/'

gan_map_file =  data_folder + '6n18_gan_mov.mrc'
pdb_file =  data_folder + 'pdb_cube.pdb'

runCommand('close all')

prot = chimera.openModels.open(pdb_file,type='PDB')[0]
gan_map = VolumeViewer.volume.open_volume_file(gan_map_file)[0]

low_lim = prot.atoms[0].coord()
high_lim = prot.atoms[7].coord()

#interploate
points_to_interp = [np.array(at.coord()) for at in prot.atoms]
data_interp = gan_map.interpolated_values(points_to_interp)

print(gan_map.full_matrix().shape)
new_box = cut_box_gx(gan_map, low_lim,high_lim)
print(new_box.shape)
print(gan_map.full_matrix().shape)
