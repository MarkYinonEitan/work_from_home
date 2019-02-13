import sys
eman2_path = '/specific/netapp5_2/iscb/wolfson/Mark/Tools/eman2-master/programs/'
sys.path.insert(0, eman2_path)
import dbcreator
reload(dbcreator)
from dbcreator import DBcreator
from dbcreator import LabelbyAAType
from dbcreator import BoxCenterAtCG
from glob import glob1
import os

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()

from process_rotamers_data import get_mrc_file_name, get_pdb_id
from process_rotamers_data import read_rotamers_data_text_file

resolution = 2.4
apix = 1


general_mrc_file_name = get_mrc_file_name('XXXX',resolution,apix)
file_names_suffix = general_mrc_file_name[5:-4]
input_pdb_folder = dir_path+'/../data/rotamersdata/pdbs_H_added/'
mrc_maps_folder = dir_path+'/../data/rotamersdata/mrcs/MRCs_' + file_names_suffix
database_folder = dir_path+'/../data/rotamersdata/DB_'+ file_names_suffix +'/'
database_file_name = ""
list_file_name = dir_path+'/../data/rotamersdata/DatasetForBBDepRL2010.txt'
label_display_file = dir_path+'/../data/temp/labels.txt'

new_db = DBcreator( input_pdb_folder, mrc_maps_folder,database_folder,database_file_name,
     list_file_name,resolution,apix=apix,label=LabelbyAAType, box_center=BoxCenterAtCG,
     cubic_box_size = 11)

new_db.print_labels_to_file(label_display_file)
print new_db.label_statistics

#get list of pdbs
pdbs_in_pdb_files = set([get_pdb_id(x) for x in glob1(input_pdb_folder,'*.pdb')])
pdbs_in_mrc_files = set([get_pdb_id(x) for x in glob1(mrc_maps_folder,'*.mrc' )])
pdbs_in_list = read_rotamers_data_text_file(list_file_name).keys()


list_of_pdb_ids = list(pdbs_in_pdb_files.intersection(pdbs_in_mrc_files).intersection(pdbs_in_list))

for pdb_id in list_of_pdb_ids:
    file_to_save =  database_folder + pdb_id+'_'+file_names_suffix +'.pkl.gz'
    if os.path.isfile(file_to_save):
	continue
    training_pairs, debug_data=new_db.calc_all_training_pairs_for_pdb(pdb_id)
    new_db.save_training_pairs(training_pairs, debug_data,filename =file_to_save )








