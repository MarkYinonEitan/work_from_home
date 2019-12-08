import os,sys
from chimera import runCommand

python_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/python/'
chimera_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera/'
sys.path.append(python_path)
sys.path.append(chimera_path)


from dataset_loader import VOX_SIZE, RESOLUTION,NBOX_IN, N_SAMPLS_FOR_1V3
from createDB import calc_all_matrices,save_matrc_to_folder
#

def old_fun():
    vox_folder = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/vox6n18/'
    pdb_id = '6nt8'
    pdb_file = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8.pdb'
    map_file = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8_molmap.mrc'
    res = 6.6
    vx_size = 2.0
    inp_mtrc, output_mtrx, local_fit_matrix = calc_all_matrices(pdb_file, map_file,vx_size = vx_size, res = res)
    #save data to folder
    save_matrc_to_folder(vox_folder,pdb_id,inp_mtrc, output_mtrx, local_fit_matrix)

if __name__ == "chimeraOpenSandbox":
    print "Creating INPUT DATA"
    pdb_id = sys.argv[3]
    pdb_file = sys.argv[4]
    map_file = sys.argv[5]
    vox_folder = sys.argv[6]
    vx_size = float(sys.argv[7])
    res = float(sys.argv[8])

    inp_mtrc, output_mtrx, local_fit_matrix = calc_all_matrices(pdb_file, map_file,vx_size = vx_size, res = res)
    print("INPUTS CREATED")
    save_matrc_to_folder(vox_folder,pdb_id,inp_mtrc, output_mtrx, local_fit_matrix)
    print("INPUTS SAVED")
    runCommand('stop')
