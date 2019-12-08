import os,sys

if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/python/'
chimera_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/chimera/'
sys.path.append(python_path)
sys.path.append(chimera_path)


from dataset_loader import VOX_SIZE, RESOLUTION,NBOX_IN, N_SAMPLS_FOR_1V3
from createDB import calc_all_matrices,save_matrc_to_folder
#


vox_folder = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/vox6n18/'
pdb_id = '6nt8'
pdb_file = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8.pdb'
map_file = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/6nt8_molmap.mrc'


inp_mtrc, output_mtrx, local_fit_matrix = calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION)
#save data to folder
save_matrc_to_folder(vox_folder,pdb_id,inp_mtrc, output_mtrx, local_fit_matrix)
