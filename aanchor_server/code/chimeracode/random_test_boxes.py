import gzip
import cPickle
import numpy
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data

dir_path = os.path.dirname(os.path.realpath(__file__))
input_pdb_folder = dir_path+'/../data/rotamersdata/pdbs_H_added/'
file_name =  dir_path+'/../data/rotamersdata/DB_res30apix10/1a4i_res30apix10.pkl.gz'

pdbs_folder = ''
N_test_per_pdb = 10
pdb_id = '1a4i'
box_size = 11
apix =1.0


def get_n_random_numbers(start, end, n):
    a = range(start, end+1)
    numpy.random.shuffle(a)
    n = min(len(a),n)
    return a[0:n]

def atomsList2spec(atomsList):
    spec = ""
    for atom in atomsList:
        mod = str(atom.molecule.id);
        sub_mod = str(atom.molecule.subid);
        chainId = atom.residue.id.chainId;
        resNum = str(atom.residue.id.position);
        atomType = atom.name
        altLoc = atom.altLoc

        spec = spec + ' ' +'#' +mod +'.' + sub_mod + ':' + resNum + '.' + chainId + '@' + atomType + '.' + altLoc

    return spec

def load_images_to_chimera(box_data, db_data,pdbs_folder,apix):
    pdb_id  = db_data["pdb_id"]
    chainId = db_data["chainId"]
    resnum = db_data["resnum"]
    center_xyz = db_data["center"]
    origin_xyz = (center_xyz[0]-box_data.shape[0]/2*apix,center_xyz[1]-box_data.shape[1]/2*apix,center_xyz[2]-box_data.shape[2]/2*apix)
            
    #close all
    runCommand('close all')
    #load and display pdb 
    pdb_file = glob1(pdbs_folder,pdb_id+'*.pdb')[0]
    prot = chimera.openModels.open(pdbs_folder+pdb_file,'PDB')[0]
    runCommand('~ribbon')
    atoms_to_show = list(filter(lambda  at: (at.residue.id.position == res_data['resnum'])& (at.residue.id.chainId== res_data['chainId']), prot.atoms))
    runCommand('show ' + atomsList2spec(atoms_to_show))
    #create map
    # Create and display volume model.
    grid = Array_Grid_Data(box_data,origin = origin_xyz,step = (apix,apix,apix))
    v = volume_from_grid_data(grid, 'surface')

    
    
    return v
# -----------------------------------------------------------------------------


#def show_only_this_residue
#load file
f = gzip.open(file_name, 'rb')
training_data, debug_data =  cPickle.load(f)
f.close()

#select randomly n rows from debug data
test_indexes = get_n_random_numbers(0,len(debug_data)- 1,N_test_per_pdb)
#loop on rows



for test_ind in test_indexes:
    test_ind = 10
    box_data = numpy.reshape(training_data[test_ind][0],[box_size,box_size,box_size])
    res_data = debug_data[test_ind]
    v = load_images_to_chimera(box_data, res_data,input_pdb_folder,apix)
