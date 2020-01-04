import chimera
import os 
import sys
from chimera import runCommand

#get current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path+'\..\pythoncode')
from process_rotamers_data import read_rotamers_data_text_file



#input list file
list_file = dir_path+'/../data/rotamersdata/DatasetForBBDepRL2010.txt'
#target_folder
target_PDB_folder = dir_path+'/../data/rotamersdata/pdbs_H_added/'
existing_pdbs = os.listdir(target_PDB_folder)

#load rotamers data
all_rotamers_data = read_rotamers_data_text_file(list_file)
all_pdbs = all_rotamers_data.keys()

problematic_pdbs = [];

for pdb_id in all_pdbs:
    if (pdb_id+ '_H.pdb') in existing_pdbs:
        continue
    try:
        #delete
        runCommand('delete')
        #load from pdb
        prot = chimera.openModels.open(pdb_id,'PDB')[0]
        #add hydrogens
        runCommand('addh')
        #save
        runCommand("write 0 " + target_PDB_folder+pdb_id+'_H.pdb')
    except:
        problematic_pdbs.append(pdb_id)
        



