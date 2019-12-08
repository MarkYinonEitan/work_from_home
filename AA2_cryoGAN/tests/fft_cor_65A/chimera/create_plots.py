import chimera
from chimera import specifier
import VolumeViewer
from Matrix import euler_xform
from chimera import runCommand
import numpy
from numpy.fft import fftn, ifftn
from numpy import conj
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


real_map_file = '/Users/markroza/Documents/GitHub/work_from_home/test_sim_1/data/emd_0505.map'
mol_map_file = '/Users/markroza/Documents/GitHub/work_from_home/test_sim_1/data/6nt8_molmap.mrc'
gan_map_file = '/Users/markroza/Documents/GitHub/work_from_home/test_sim_1/data/6n18_gan_mov.mrc'
pdb_file = '/Users/markroza/Documents/GitHub/work_from_home/test_sim_1/data/6nt8.pdb'
pdb_frame_file = '/Users/markroza/Documents/GitHub/work_from_home/test_sim_1/data/pdb_xyz_frame.pdb'
dr=1
central_pos = numpy.array([112.389, 86.11, 119.811])
step_df =(dr,dr,dr)# step is default to (1,1,1)
small_size = 10
large_size = 160


def fft_corr(large_map, small_map):
	a1_f = conj(fftn(small_map,s=large_map.shape))
	a2_f = fftn(large_map,s=large_map.shape)
	a3_f = a1_f*a2_f
	corr_map = numpy.abs(ifftn(a3_f))
	corr_map[:] = corr_map[:]/numpy.std(corr_map[:])*0.5
	corr_map[:] = corr_map[:] - numpy.mean(corr_map[:])+0.5

	opt_place_temp = numpy.where(corr_map == numpy.amax(corr_map))
	opt_place = numpy.array([opt_place_temp[0][0],opt_place_temp[1][0],opt_place_temp[2][0]])
	return opt_place, corr_map

def bounds(center, size):
    #takes the coord of the center of res, and produce the
    # min,max of the box
    (x,y,z) = center
    d = size/2
    minxyz =(x-d, y-d, z-d)
    maxxyz =(x+d, y+d, z+d)
    return (minxyz,maxxyz,step_df)

def cut_box(full_map, center, size):
    ijk_center = full_map.data.xyz_to_ijk(center)
    #convert the xyz_coordinate to ijk_matrix coordinate
    ijk_bound = bounds(ijk_center,size)
    #ijk_bound is the limit of the 10*10*10 box we want to extract
    #ijk_bound is a tupe in the form (minxyz,maxxyz,step)
    full_map.new_region(ijk_min = ijk_bound[0], ijk_max = ijk_bound[1],
                        ijk_step = ijk_bound[2], show = True)
    #cut the full_map to a map of the box's region
    mx =  full_map.matrix()

    mx[:] = mx[:]/numpy.std(mx[:])*0.5
    mx[:] = mx[:] - numpy.mean(mx[:])+1


    return mx


def get_frame_in_position():
    prot = chimera.openModels.open(pdb_frame_file,type='PDB')[0]

    for at in prot.atoms:
        at.setCoord(chimera.Point(at.coord().x+central_pos[0],at.coord().y+central_pos[1],at.coord().z+central_pos[2]))

    return        




def whole_map_with_the_box():
    runCommand('close all')
    real_map = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    real_to_cut = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    real_small = cut_box(real_to_cut, central_pos, small_size)
    get_frame_in_position()

    print('WHOLE PICTURE')
    print('1. Show Spheres in PDB')
    print('2.make small map look like a cube and take a picture')
    print('3 Orient and save')
    print('4 Remove Pdb')
    print('5 Save')




def box_from_real_map():
    runCommand('close all')
    real_map = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    real_to_cut = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    real_small_1 = cut_box(real_to_cut, central_pos, small_size)
    real_small_2 = cut_box(real_map, central_pos, small_size)
    get_frame_in_position()

    print('REAL')
    print('1. Show Spheres in PDB')
    print('2.make one map look like a cube (surface) and other a mesh')
    print('3 Orient and save')
    print('4 Remove Pdb')
    print('5 Save')
    


def box_from_gan_map():
    runCommand('close all')
    gan_to_cut_1 = VolumeViewer.volume.open_volume_file(gan_map_file)[0]
    gan_small_1 = cut_box(gan_to_cut_1, central_pos, small_size)
    gan_to_cut_2 = VolumeViewer.volume.open_volume_file(gan_map_file)[0]
    gan_small_2 = cut_box(gan_to_cut_2, central_pos, small_size)
    real_to_cut = VolumeViewer.volume.open_volume_file(real_map_file)[0]
    real_small_1 = cut_box(real_to_cut, central_pos, small_size)

    get_frame_in_position()

    print('GAN')
    print('1. Show Spheres in PDB')
    print('2.make one map look like a cube (surface) and other a mesh')
    print('3 Orient and save')
    print('4 Remove Pdb')
    print('5 Save')


#Whole map  with the box
runCommand('close all')
gan_to_cut = VolumeViewer.volume.open_volume_file(gan_map_file)[0]
real_to_cut = VolumeViewer.volume.open_volume_file(real_map_file)[0]
mol_to_cut = VolumeViewer.volume.open_volume_file(mol_map_file)[0]
real_map = VolumeViewer.volume.open_volume_file(real_map_file)[0]

#cut
gan_small = cut_box(gan_to_cut, central_pos, small_size)
real_small = cut_box(real_to_cut, central_pos, small_size)
mol_small = cut_box(mol_to_cut, central_pos, small_size)
mx_large = cut_box(real_map, central_pos, large_size)

whole_map_with_the_box()

box_from_real_map()
box_from_gan_map()
