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



def test_results(out_pdb_file, ref_pdb_file , out_file = 't1.txt'):
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
    with open(out_file, 'w') as f:
        for residue_type in res_types:
            out_residues = filter(lambda x: x.type==residue_type, out_prot.residues )
            print(residue_type)
            n_true=0
            n_false=0
            f.write(residue_type+'\n')
            for res in out_residues:
                atm = res.findAtom('K')
                position = atm.coord()
                f.write(str(atm.residue.id.position) + ' : ' +str(atm.bfactor)+' ### ')
                inxs, ress = kdt4.in_range(position,THR_DIST)
                if len(inxs) !=1:
                    f.write( '  FALSE Num Near{}'.format(len(inxs)))
                    n_false+=1
                else:
                    res_found = ress[0]
                    if res_found.type != residue_type:
                        n_false+=1
                        f.write( '  FALSE UNCORRECT TYPE :{}'.format(res_found.type))

                    else:
                        f.write( '  TRUE ' +res_found.id.chainId + ' ' +str(res_found.id.position) )
                        n_true+=1
                f.write('  ' + str(position)+'\n')
            f.write('Summary Confidence  ' + str(n_true/(n_true+n_false+0.0001))+'\n')
            print('Conf', n_true/(n_true+n_false+0.0001))

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
        test_results(out_file, ref_file , out_file = results_file)

    if n_command == 'test_one':

        out_file = sys.argv[4]
        ref_file = sys.argv[5]
        res_type = sys.argv[6]
        res_thr = float(sys.argv[7])

        test_one_label(out_file, ref_file , res_type, res_thr)
    runCommand('stop')








## Statistcs by label

## Show
