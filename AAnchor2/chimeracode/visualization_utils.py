import chimera

import VolumeViewer
import VolumeData
from VolumeData import Array_Grid_Data
from VolumeData import Grid_Data
import MoleculeMap
from MoleculeMap import molecule_map
import numpy as np
import os, sys
import MarkChimeraUtils
from chimera import runCommand


if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../pythoncode/utils/'
sys.path.append(utils_path)



def create_map_from_box(box, box_center=(0,0,0), apix = 1):
    #takes the path of a matrix and shows it in chimera
    data = Array_Grid_Data(box, step = (apix, apix,apix), origin = box_center)
    vmap = VolumeViewer.Volume(data)
    vmap.show()

def visual_box_test(data_dict, pdb_file, half_bx_size=5):

    runCommand('close all')

    CA_list=[]
    prot = chimera.openModels.open(pdb_file,'PDB')[0]
    #load pdb file
    for item in data_dict:
        center = item["ref_data"]['CG_pos']
        bx_not_sw = item["box"]
        bx  = np.swapaxes(bx_not_sw,0,2)
        center_corrected = (center[0]-half_bx_size, center[1]-half_bx_size, center[2]-half_bx_size)
        create_map_from_box(bx, box_center=center_corrected, apix = 1)
        res = MarkChimeraUtils.getResByNumInChain(prot, item["ref_data"]['chainId'],item["ref_data"]['pos'])
        CA_list.append(res.findAtom('CA'))

    atoms_spec = MarkChimeraUtils.atomsList2spec(CA_list)

    runCommand("delete {} za>8".format(atoms_spec))
