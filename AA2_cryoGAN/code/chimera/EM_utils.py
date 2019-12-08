import chimera
from chimera import specifier
import VolumeViewer
from Matrix import euler_xform
from chimera import runCommand
import numpy as np
from numpy.fft import fftn, ifftn
from numpy import conj

def get_box(v,box_xyz_grid):
    assert box_xyz_grid[0].shape == box_xyz_grid[1].shape
    assert box_xyz_grid[1].shape == box_xyz_grid[2].shape

    points =  zip(np.reshape(box_xyz_grid[0],-1),np.reshape(box_xyz_grid[1],-1),np.reshape(box_xyz_grid[2],-1))
    values = v.interpolated_values(points)
    box = np.reshape(values, box_xyz_grid[1].shape)

    return box

def calc_box_grid(center, size, apix=1):

    x=float(center[0])
    y=float(center[1])
    z=float(center[2])

    cbs = float(size)
    apx = float(apix)
    x_gr = np.linspace(x-cbs/2,x+cbs/2,cbs)
    y_gr = np.linspace(y-cbs/2,y+cbs/2,cbs)
    z_gr = np.linspace(z-cbs/2,z+cbs/2,cbs)
    X,Y,Z = np.meshgrid(x_gr,y_gr,z_gr,indexing = 'ij')
    box_xyz = (X,Y,Z)

    return box_xyz

def cut_box_gx(full_map, low_bound,high_bound,abs_step= [1,1,1]):

    ijk_low = full_map.data.xyz_to_ijk(low_bound)
    #convert the xyz_coordinate to ijk_matrix coordinate
    ijk_high= full_map.data.xyz_to_ijk(high_bound)
    #ijk_bound is the limit of the 10*10*10 box we want to extract
    #ijk_bound is a tupe in the form (minxyz,maxxyz,step)
    full_map.new_region(ijk_min = ijk_low, ijk_max = ijk_high,
                        ijk_step=abs_step , show = False)

    mx = full_map.matrix()
    return mx

def assert_all_maps():
    for mp in VolumeViewer.volume_list():
        assert_map_obj(mp)
    return


def assert_map_obj(map_obj):
    xyz1 = map_obj.ijk_to_global_xyz((0,0,0))
    for x in xyz1:
        if np.abs(round(x)-x)>0.01:
            raise NameError('Map Grid is Not Integer with step 1: Use make_map_integer')
    xyz2 = map_obj.ijk_to_global_xyz((1,1,1))
    dxyz = np.array(xyz2)-np.array(xyz1)
    for d in dxyz:
        if abs(abs(d)-1)>0.0001:
            raise NameError('Map Grid is Not Integer with step 1: Use make_map_integer')
    return

def interp_map_to_good_grid(mp):
    new_map_id = mp.id+100

    #create_grid
    low_bound = mp.ijk_to_global_xyz((0,0,0))
    high_bound = mp.ijk_to_global_xyz(mp..full_matrix().shape)

    low_bound_int = np.floor(np.array([min(low_bound[0],high_bound[0]),min(low_bound[1],high_bound[1]),min(low_bound[2],high_bound[2])]))

    high_bound_int = np.floor(np.array([max(low_bound[0],high_bound[0]),max(low_bound[1],high_bound[1]),max(low_bound[2],high_bound[2])]))

    dN = high_bound_int-low_bound_int
    Nx = dN[0]
    Ny = dN[1]
    Nz = dN[2]








    #temp map for grid
    vop new  grid_only   size  {Nx},{Ny},{Nz}  gridSpacing 1.0  origin ox,oy,zz ] [ cellAngles α,β,γ ] [ valueType value-type ] [ modelId  N ]
    .format(Nx,Ny,Nz,ox,oy,oz)

    return new_map
