import chimera
from chimera import runCommand
import glob




def create_data_set(db_folder, min_res, max_res, list_file = "list.txt"):

    a = 5
    return 0


list_file = "list.txt"

db_folder = '/Users/markroza/Documents/work_from_home/NNcourse_project/data/res6/synth/'

min_res = 5.25
max_res = 6.25
pdb_files = glob.glob1(db_folder,'*.pdb')

synth_map_id = 1001

map_num=9999

list_file = db_folder+list_file

with open(list_file,"w") as f_list:
    f_list.write('PDB_ID MAP_ID RESOLUTION\n')


for pdb_file in pdb_files:

    runCommand('close all')
    pdb_code = pdb_file[0:4]

    #open and add H
    prot1 = chimera.openModels.open(db_folder+pdb_file)[0]
    pdb_id = prot1.id
    runCommand('addh spec #{}'.format(pdb_id))
    #randomize resolution
    res = np.random.uniform(min_res,max_res)
    res = np.round(res,decimals =1)
    vx_size  = np.random.uniform(res/4.0,res/2.5)
    #create map
    runCommand('molmap #{} {} gridSpacing {} modelId {}'\
               .format(pdb_id, res, vx_size,synth_map_id))
    map_num = map_num-1
    #save map
    runCommand('volume #{} save {}'.format(synth_map_id,db_folder+'emd-'+str(map_num)+'.mrc'))
    with open(list_file,"a") as f_list:
        f_list.write('{} {} {}\n'.format(pdb_code,map_num,res))


               



    



