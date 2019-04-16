import chimera
from chimera import runCommand
import glob
import numpy as np


def read_resolution(line):
    words = line.split()
    res_word = filter(lambda x: x[-2:]=="\xc3\x85",words)[0]
    return float(res_word[:-2])

def save_files(emdb_list_file, nums, save_folder):
    trips = read_list_file(emdb_list_file)
    out_file = save_folder +'\list.txt'

    with open(out_file,"w") as f_list:
        f_list.write('PDB_ID MAP_ID RESOLUTION\n')
    for n in nums:
        emdb_id = trips[n][0]
        pdb_id = trips[n][1]
        res = trips[n][2]
        runCommand('close all')
        runCommand('open pdbID:{}'.format(pdb_id))
        runCommand('write  0 ' + save_folder+ "/{}.pdb".format(pdb_id))
        runCommand('close all')
        runCommand('open emdbID:{}'.format(emdb_id))
        runCommand('volume #0 save ' + save_folder+ "/emd-{}.mrc".format(emdb_id))
        with open(out_file,"a") as f_list:
            f_list.write('{} {} {}\n'.format(pdb_id,emdb_id,res))

    return


def read_list_file(emdb_list_file):
    trips = []
    with open(emdb_list_file) as fp:
        fp.readline()
        for cnt, line in enumerate(fp):
            emdb_num = line[4:8]
            pdb_id = line[-6:-2]
            res = read_resolution(line)
            trips.append((emdb_num,pdb_id,res))
    return trips

pdb_model = 1001
emd_model = 3001

emdb_list_file = "/Users/markroza/Documents/GitHub/work_from_home/NNproject/data/emdbdumps_res6.txt"
show_num = 29
save_nums = [10,11,15,16,18,26,29]

needs_sym = [12,13,19]

trips = read_list_file(emdb_list_file)
emdb_id = trips[show_num][0]
pdb_id = trips[show_num][1]

runCommand('close all')
runCommand('open pdbID:{}'.format(pdb_id))
runCommand('open emdbID:{}'.format(emdb_id))
print(show_num, trips[show_num][2])

save_files(emdb_list_file, save_nums, '/Users/markroza/Documents/temp/res6/')
