import sys
from glob import glob1
import os
import multiprocessing
import re
import threading	

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../utils/'
sys.path.append(utils_path)
eman2_path = '/specific/netapp5_2/iscb/wolfson/Mark/Tools/eman2-master/programs/'
sys.path.insert(0, eman2_path)


import dbcreator
reload(dbcreator)
from dbcreator import DBcreator
from dbcreator import LabelbyAAType
from dbcreator import BoxCenterAtCG


from process_rotamers_data import get_mrc_file_name, get_pdb_id
from process_rotamers_data import read_rotamers_data_text_file


def create_db_from_n_to_m(n=0,m=10**10):

    #parameters
    resolution = 3
    apix_in = -0.1
    apix_out = 1


    thread_num = 50

    data_folder = dir_path+'/../../data/'

    general_mrc_file_name = get_mrc_file_name('XXXX',resolution,apix_in)
    file_names_suffix = general_mrc_file_name[5:-4]
    input_pdb_folder = data_folder+'/rotamersdata/pdbs_H_added/'
    mrc_maps_folder = data_folder+'/rotamersdata/mrcs/MRCs_' + file_names_suffix
    database_folder = data_folder+'/rotamersdata/DB_'+ file_names_suffix +'/'
    database_file_name = ""
    list_file_name = data_folder+'/rotamersdata/DatasetForBBDepRL2010.txt'
    label_display_file = data_folder+'/temp/labels.txt'

    new_db = DBcreator( input_pdb_folder, mrc_maps_folder,database_folder,database_file_name,
         list_file_name,resolution,apix=apix_out,label=LabelbyAAType, box_center=BoxCenterAtCG,
         cubic_box_size = 11)

    new_db.print_labels_to_file(label_display_file)
    print new_db.label_statistics

    #get list of pdbs
    pdbs_in_pdb_files = set([get_pdb_id(x) for x in glob1(input_pdb_folder,'*.pdb')])
    pdbs_in_mrc_files = set([get_pdb_id(x) for x in glob1(mrc_maps_folder,'*.mrc' )])
    pdbs_in_list = read_rotamers_data_text_file(list_file_name).keys()


    list_of_pdb_ids = list(pdbs_in_pdb_files.intersection(pdbs_in_mrc_files).intersection(pdbs_in_list))
    
    ## run only on selected pdbs
    list_of_pdb_ids= list_of_pdb_ids[n:m]


    db_create_classes = []		
    for r in range(thread_num):		
        new_db = DBcreator( input_pdb_folder, mrc_maps_folder,database_folder,database_file_name, list_file_name,resolution,apix=apix_out,label=LabelbyAAType, box_center=BoxCenterAtCG, cubic_box_size = 11)
        db_create_classes.append(new_db)

    #run in threads
    list_of_pdbs_ids_threads = [list_of_pdb_ids[x:x+thread_num] for x in range(0,len(list_of_pdb_ids),thread_num)]

    for pdbs_list in list_of_pdbs_ids_threads:
        threads=[]
        for db_crt, pdb_id in zip(db_create_classes,pdbs_list):
	        file_to_save =  database_folder + pdb_id+'_'+file_names_suffix +'.pkl.gz'
	        mrc_file = glob1(mrc_maps_folder,pdb_id+'*.mrc' )[0]
	        t = threading.Thread(target=db_crt.calc_and_save_all_training_pairs_for_pdb, args = (pdb_id,mrc_file,file_to_save))
	        threads.append(t)
	        t.start()

        for t in threads:
        	t.join()


if __name__ == "__main__":
    n_start = int(sys.argv[1])
    n_end = int(sys.argv[2])
    create_db_from_n_to_m(n=n_start,m=n_end)


    #for pdb_id in list_of_pdb_ids:
    #    file_to_save =  database_folder + pdb_id+'_'+file_names_suffix +'.pkl.gz'
    #    mrc_file = glob1(mrc_maps_folder,pdb_id+'*.mrc' )[0]
    #    if os.path.isfile(file_to_save):
    #	continue
    #    training_pairs, debug_data=new_db.calc_all_training_pairs_for_pdb(pdb_id,mrc_file_name = mrc_file)
    #    new_db.save_training_pairs(training_pairs, debug_data,filename =file_to_save )








