import numpy as np
import sys

nets_path = '/specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/AAnchor2/pythoncode/nets/'
sys.path.append(nets_path)

try:
  import all_nets
except ImportError:
  print ("RUN without TENSORFLOW")

def assert_vx_size_and_resolution(vx_size,res):
    if (vx_size != dataset_loader.VOX_SIZE ) or (res != dataset_loader.RESOLUTION ):
        raise Exception("VX_SIZE or RES uncorrect")


def read_list_file(list_file):
    pairs=[]
    with open(list_file) as fp:
        line = fp.readline()#read header
        line = fp.readline()
        while line:
            wrds = line.split()
            pdb_id = wrds[0]
            emd_id = wrds[1]
            res = float(wrds[2])
            train_test = wrds[3]
            is_virus = wrds[4]

            line = fp.readline()
            pairs.append({"pdb_file":pdb_id,"emd_file":emd_id,"res":res,"train_test":train_test, "is_virus":is_virus})
    return pairs

def read_thr_file(thr_file):

    thr_dict = {}
    with open(thr_file) as fp:
        line = fp.readline()
        while line:
            wrds = line.split()
            res_type = wrds[0]
            thr = float(wrds[1])

            line = fp.readline()
            thr_dict[res_type] = thr

    return thr_dict

def get_net_by_string(net_string):
    if net_string == 'V5_no_reg':
        return all_nets.V5_no_reg()
    if net_string == 'V5_DROP_REG':
        return all_nets.V5_Drop_Reg()
    if net_string == 'V5_DROP_REG_2':
        return all_nets.V5_Drop_Reg_2()
    if net_string == 'V5_REG_3':
        return all_nets.V5_Reg_3()

    raise Exception('No Net Found')
