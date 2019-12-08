import chimera
from chimera import specifier
import VolumeViewer
from Matrix import euler_xform
from chimera import runCommand
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def get_non_zero_grid(input_map,limits_in,limits_out):
    all_bounds = input_map.xyz_bounds()
    full_map = input_map.full_matrix()

    x_min =  0
    x_max =  full_map.shape[0]
    while (np.sum(full_map[x_min+1,:,:]) ==0.0):
        x_min +=1
    while (np.sum(full_map[x_max-2,:,:]) ==0.0):
        x_max -=1

    y_min =  0
    y_max =  full_map.shape[1]
    while (np.sum(full_map[:,y_min+1,:]) ==0):
        y_min +=1
    while (np.sum(full_map[:,y_max-2,:])==0):
        y_max -=1

    z_min =  0
    z_max =  full_map.shape[2]
    while (np.sum(full_map[:,:,z_min+1]) ==0):
        z_min +=1
    while (np.sum(full_map[:,:,z_max-2]) ==0):
        z_max -=1

    x_min_A,y_min_A,z_min_A = input_map.ijk_to_global_xyz((x_min,y_min,z_min))
    x_max_A,y_max_A,z_max_A = input_map.ijk_to_global_xyz((x_max,y_max,z_max))

    x_min_A = np.floor(np.clip(x_min_A,limits_out[0][0],limits_in[0][0]))
    y_min_A = np.floor(np.clip(y_min_A,limits_out[1][0],limits_in[1][0]))
    z_min_A = np.floor(np.clip(z_min_A,limits_out[2][0],limits_in[2][0]))

    x_max_A = np.ceil(np.clip(x_max_A,limits_in[0][1],limits_out[0][1]))
    y_max_A = np.ceil(np.clip(y_max_A,limits_in[1][1],limits_out[1][1]))
    z_max_A = np.ceil(np.clip(z_max_A,limits_in[2][1],limits_out[2][1]))


    return x_min_A,x_max_A,y_min_A,y_max_A,z_min_A,z_max_A


def resample_map_with_chimera(initial_map,limits_in=([10.0**6,-10.0**6],[10.0**6,-10.0**6],[10.0**6,-10.0**6]),limits_out=([-10.0**6,10.0**6],[-10.0**6,+10.0**6],[-10.0**6,+10.0**6])):

        assert initial_map.id == 17

        #correct limits
        x_out_min = min(limits_out[0][0],limits_in[0][0])
        y_out_min = min(limits_out[1][0],limits_in[1][0])
        z_out_min = min(limits_out[2][0],limits_in[2][0])
        x_out_max = max(limits_out[0][1],limits_in[0][1])
        y_out_max = max(limits_out[1][1],limits_in[1][1])
        z_out_max = max(limits_out[2][1],limits_in[2][1])
        limits_out = ([x_out_min,x_out_max],[y_out_min,y_out_max],[z_out_min,z_out_max])

        x_min,x_max,y_min,y_max,z_min,z_max = get_non_zero_grid(initial_map,limits_in,limits_out)

        x_grid = np.arange(x_min-apix,x_max+apix,apix)
        y_grid = np.arange(y_min-apix,y_max+apix,apix)
        z_grid =  np.arange(z_min-apix,z_max+apix,apix)

        x, y, z = np.meshgrid(x_grid, x_grid, x_grid)

        origin_xyz = (x_min,y_min,z_min)

        #move origin to cornerd

        new_map_command = 'vop new  int_map  size  %d,%d,%d ' %(len(x_grid),len(y_grid),len(z_grid))
        new_map_command = new_map_command + ' gridSpacing %s ' %apix
        new_map_command = new_map_command + 'origin  %s,%s,%s' %(origin_xyz[0],origin_xyz[1],origin_xyz[2])
        new_map_command = new_map_command + ' modelId  27'
        runCommand(new_map_command)

        resample_map_command =  'vop resample  #{}  onGrid #{} modelId 37'.format(initial_map.id, '27')
        runCommand(resample_map_command)
        resampled_map = chimera.openModels.list()[2]
        for mdl in chimera.openModels.list():
                print "DEBUG model if" , mdl.id
        print(resampled_map.id)
        assert resampled_map.id == 37
        runCommand('close 27')
        return resampled_map




data_folder = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/'

inp_map = data_folder + '6nt8_molmap.mrc'
out_map = data_folder + '6nt8_molmap_1A.mrc'
apix = 1

runCommand('close all')
init_map = VolumeViewer.volume.open_volume_file(inp_map, model_id=17)[0]
new_map = resample_map_with_chimera(init_map)
runCommand('volume #{} save {}'.format(new_map.id,out_map))

