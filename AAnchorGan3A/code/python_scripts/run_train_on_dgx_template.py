import os
import sys
from glob import glob



code_path = "/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchorGan3A/code/"
utils_path = code_path + '/python/'
chimera_path = code_path + '/chimera/'
nets_path = code_path + '/python/aanchor_nets/'
sys.path.append(utils_path)
sys.path.append(nets_path)
sys.path.append(chimera_path)


import utils_project


import networkanalyser
from networkanalyser import NetworkAnalyser
import resultsplots
from resultsplots import SingleNetResults
import dbloader
import all_nets
from dbloader import LabelbyAAType

#parameters
MINI_BATCH_SIZE = 600
MINI_EPOCHS = 5

def train_network(inp_list_file, input_db_folder, out_folder, net_string, initial_weight_file, N_epoch):

    list = utils_project.read_list_file(inp_list_file)
    train_data_files=[]
    valid_data_files=[]
    for data_row in list:
        emd_name = data_row["emd_file"][:-4]
        file_names = [x[:-4] for x in glob(input_db_folder+'*{}*.csv'.format(emd_name))]
        if data_row["train_test"]=="TEST":
            valid_data_files = valid_data_files+file_names
        if data_row["train_test"]=="TRAIN":
            train_data_files = train_data_files+file_names

    ntwrk = utils_project.get_net_by_string(net_string)

    na =NetworkAnalyser(ntwrk, res_folder=out_folder, train_files=train_data_files,valid_files=valid_data_files,initial_weight_file=initial_weight_file,mini_batch_size=MINI_BATCH_SIZE)

    na.train_network(N_epoch = N_epoch)




    return

if __name__ == "__main__":

    inp_list_file  = "XXX_INPUT_LIST"
    inp_db_folder  = "XXX_INPUT_DB_FOLDER"
    out_folder     = "XXX_OUT_FOLDER"
    net_string     = "XXX_NET_STRING"
    weights_file   = "XXX_NET_WEIGHTS"
    N_epoch        = XXX_N_EPOCHS

    train_network(inp_list_file, inp_db_folder, out_folder, net_string, weights_file, N_epoch)
