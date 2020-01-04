import gzip
import cPickle
import numpy
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import open_volume_file


main_data_folder = 'C:/Mark/temp/Projects/NNcryoEM/data/rotamersdata/'

mrc_file  = main_data_folder + '/mrcs/MRCs_res30apix10/1a4i_res30apix10.mrc'
test_file  = main_data_folder + '../temp/test_file.txt'
pdb_file =  main_data_folder + '/pdbs_H_added/1a4i_H.pdb'
box_data_file = main_data_folder + 'DB_res30apix10/1a4i_res30apix10.pkl.gz'
in_test = 1
apix =1.0
#clear session
runCommand('close all')


## load files
whole_map = open_volume_file(mrc_file)[0]
prot = chimera.openModels.open(pdb_file)[0]
f = gzip.open(box_data_file, 'rb')
training_data, debug_data =  cPickle.load(f)
f.close()
res_data = debug_data[in_test]
box_data = training_data[in_test][0]
box_3d = numpy.reshape(box_data,[11,11,11])

#create small map
center_xyz = res_data["center"]
#origin_xyz = (center_xyz[0]-box_3d.shape[0]/2*apix+apix/2,center_xyz[1]-box_3d.shape[1]/2*apix+apix/2,center_xyz[2]-box_3d.shape[2]/2*apix+apix/2)
origin_xyz = (center_xyz[0]-box_3d.shape[0]/2*apix,center_xyz[1]-box_3d.shape[1]/2*apix,center_xyz[2]-box_3d.shape[2]/2*apix)
grid = Array_Grid_Data(box_3d,origin_xyz,step = (apix,apix,apix))
small_map = volume_from_grid_data(grid, 'surface')

#show only this residue
runCommand('~ribbon')
runCommand('delete element.H') #remove protons
atoms_to_show = list(filter(lambda  at: (at.residue.id.position == res_data['resnum'])& (at.residue.id.chainId== res_data['chainId']), prot.atoms))
atoms_to_show = list(filter(lambda  at: (abs(at.coord()[0] -center_xyz[0] )<5) &  (abs(at.coord()[1] -center_xyz[1] )<5) & (abs(at.coord()[2] -center_xyz[2] )<5), prot.atoms))


runCommand('show ' + atomsList2spec(atoms_to_show))
runCommand('represent bs ' + atomsList2spec(atoms_to_show))


#calc cg

atoms = list(filter(lambda  at: (at.residue.id.position == res_data['resnum'])& (at.residue.id.chainId== res_data['chainId']), prot.atoms))
X=0
Y=0
Z=0
n=0
for at in atoms:
    X = X+at.coord()[0]
    Y = Y+at.coord()[1]
    Z = Z+at.coord()[2]
    n+=1
X =X/n
Y =Y/n
Z =Z/n

print ' CG calculated : ',(X,Y,Z)
print ' CG from file : ', res_data["center"]

#check coordinates
center_ijk = (res_data['outbox_x'][5,5,5],res_data['outbox_y'][5,5,5],res_data['outbox_z'][5,5,5])
center_xyz = whole_map.ijk_to_global_xyz(center_ijk)
print 'center of the box whole_map ',  center_xyz
print 'center of the box small_map ',  small_map.ijk_to_global_xyz((5,5,5))

plus_corner_ijk = (res_data['outbox_x'][10,10,10],res_data['outbox_y'][10,10,10],res_data['outbox_z'][10,10,10])
plus_corner_xyz = whole_map.ijk_to_global_xyz(plus_corner_ijk)
print 'plus corner of the box ',  plus_corner_xyz
print 'plus corner box small_map ',  small_map.ijk_to_global_xyz((10,10,10))

print 'value at center box: ', box_3d[5,5,5]
print 'value at center whole map: ', whole_map.interpolated_values([center_xyz])[0]
print 'value at center small map: ', small_map.interpolated_values([center_xyz])[0]

print '  '
print 'value at plus corner box: ', box_3d[10,10,10]
print 'value at plus corner map: ', whole_map.interpolated_values([plus_corner_xyz])[0]
print 'value at plus corner small map: ', small_map.interpolated_values([small_map.ijk_to_global_xyz((10,10,10))])[0]


print '  '
minus_corner_ijk = (res_data['outbox_x'][0,0,0],res_data['outbox_y'][0,0,0],res_data['outbox_z'][0,0,0])
print 'value at minus corner box: ', box_3d[0,0,0]
print 'value at minus corner map: ', whole_map.interpolated_values([whole_map.ijk_to_global_xyz(minus_corner_ijk)])[0]
print 'value at plus corner small map: ', small_map.interpolated_values([small_map.ijk_to_global_xyz((0,0,0))])[0]

print '  '
test_x = 5
test_y = 5
test_z = 6
test_corner_ijk = (res_data['outbox_x'][test_x,test_y,test_z],res_data['outbox_y'][test_x,test_y,test_z],res_data['outbox_z'][test_x,test_y,test_z])
print 'value at test corner box: ', box_3d[test_x,test_y,test_z]
print 'value at test corner map: ', whole_map.interpolated_values([whole_map.ijk_to_global_xyz(test_corner_ijk)])[0]
print 'value at test corner small map: ', small_map.interpolated_values([small_map.ijk_to_global_xyz((test_x,test_y,test_z))])[0]
print 'position of the box whole_map ',  whole_map.ijk_to_global_xyz(test_corner_ijk)
print 'position of the box small_map ', small_map.ijk_to_global_xyz((test_x,test_y,test_z))


f_test = open(test_file,'w')

for in_x in range(11):
    for in_y in range(11):
        for in_z in range(11):
            f_test.write(str(in_x)+ ' , ' +str(in_y)+ ' , ' +str(in_z) + ' , ')
            ijk = (res_data['outbox_x'][in_x,in_y,in_z],res_data['outbox_y'][in_x,in_y,in_z],res_data['outbox_z'][in_x,in_y,in_z])
            box_value = box_3d[in_x,in_y,in_z]
            whole_map_value = whole_map.interpolated_values([whole_map.ijk_to_global_xyz(ijk)])[0]
            small_map_value = small_map.interpolated_values([small_map.ijk_to_global_xyz((in_x,in_y,in_z))])[0]
            f_test.write(str(box_value)+ ',' +str(whole_map_value)+ ',' +str(small_map_value) + '\n')

f_test.close()
