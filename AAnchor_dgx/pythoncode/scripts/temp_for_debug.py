# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import open_volume_file

apix = 1.0


runCommand('close all')

map_file = "/specific/netapp5_2/iscb/wolfson/Mark/Projects/NNcryoEM/data/cryoEM/raw_data/maps/emd-3039.map"
whole_map = open_volume_file(map_name,model_id=17)[0]



all_bounds = whole_map.xyz_bounds()


x_min =  all_bounds[0][0]
y_min =  all_bounds[0][1]
z_min =  all_bounds[0][2]

x_max =  all_bounds[1][0]
y_max =  all_bounds[1][1]
z_max =  all_bounds[1][2]

cubic_box_size=(11,11,11)

x_start = x_min + float(cubic_box_size[0])*apix/2
x_end = x_max - float(cubic_box_size[0])*apix/2
y_start = y_min + float(cubic_box_size[1])*apix/2
y_end = y_max - float(cubic_box_size[1])*apix/2
z_start = z_min + float(cubic_box_size[2])*apix/2
z_end = z_max - float(cubic_box_size[2])*apix/2

x_centers = np.arange(x_start,x_end,apix)
y_centers = np.arange(y_start,y_end,apix)
z_centers =  np.arange(z_start,z_end,apix)

x_grid = np.arange(x_min,x_max,apix)
y_grid = np.arange(y_min,y_max,apix)
z_grid =  np.arange(z_min,z_max,apix)

x, y, z = np.meshgrid(x_grid, x_grid, x_grid)

#Nx,Ny,Nz=whole_map.matrix().shape
origin_xyz = whole_map.data_origin_and_step()[0]
new_map_command = 'vop int_map new   size  %d,%d,%d ' %(len(x_grid),len(y_grid),len(z_grid))
new_map_command = new_map_command + ' gridSpacing %s ' %apix
new_map_command = new_map_command + 'origin  %s,%s,%s' %(origin_xyz[0],origin_xyz[1],origin_xyz[2])
new_map_command = new_map_command + ' modelId  27'
#cellAngles α,β,γ ] [ valueType value-type ]
runCommand(new_map_command)

resample_map_command =  'vop resample  {}  onGrid {}'.format(whole_map.name, 'int_map')# [ boundingGrid true|false ] [ gridStep N | Nx,Ny,Nz ] [ gridSubregion name | i1,j1,k1,i2,j2,k2 | all ] [ valueType value-type ]  general-options
runCommand(resample_map_command)





map_resampled = np.zeros(x.shape)
k=0
for in_x in range(len(x_grid)):
    for in_y in range(len(y_grid)):
        for in_z in range(len(z_grid)):
            map_resampled[in_x,in_y,in_z] = mrc_map.sget_value_at_interp(x_grid[in_x],y_grid[in_y],z_grid[in_z])
            k =k+1
            if k%100000 == 0:
                print k
                print x_grid[in_x],y_grid[in_y],z_grid[in_z]
                print map_resampled[in_x,in_y,in_z]
                print x[in_x,in_y,in_z],y[in_x,in_y,in_z] ,z[in_x,in_y,in_z]


print finished
