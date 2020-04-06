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

cur_pass = os.path.dirname(os.path.realpath(__file__))

utils_path = cur_pass + '/../pythoncode/utils/'
chimera_path = cur_pass + '/../chimeracode/'
sys.path.append(utils_path)
import dbloader


def create_map_from_box(box, box_center=(0,0,0), apix = 1):
    #takes the path of a matrix and shows it in chimera
    data = Array_Grid_Data(box, step = (apix, apix,apix), origin = box_center)
    vmap = VolumeViewer.Volume(data)
    vmap.show()

def visual_box_test(file_name, pdb_file, N = 5, half_bx_size=5):

    runCommand('close all')

    data_dict = {}
    dbloader.load_train_data_to_dict([file_name],data_dict)

    CA_list=[]
    prot = chimera.openModels.open(pdb_file,'PDB')[0]

    in_show = set([np.random.choice(range(len(data_dict["boxes"]))) for x in range(N)])

    #load pdb file
    for it in in_show:
        ref_data = data_dict["data"][it]
        bx_not_sw = data_dict["boxes"][it]
        center = (ref_data['box_center_x'], ref_data['box_center_y'],ref_data['box_center_z'])
        bx  = np.swapaxes(bx_not_sw,0,2)
        center_corrected = (center[0]-half_bx_size, center[1]-half_bx_size, center[2]-half_bx_size)
        create_map_from_box(bx, box_center=center_corrected, apix = 1)
        res = MarkChimeraUtils.getResByNumInChain(prot, ref_data['chainId'],ref_data['pos'])
        CA_list.append(res.findAtom('CA'))

    atoms_spec = MarkChimeraUtils.atomsList2spec(CA_list)

    runCommand("delete {} za>8".format(atoms_spec))
