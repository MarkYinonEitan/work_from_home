import numpy
import chimera
from glob import glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import open_volume_file

map_name = 'C:/Mark/temp/Projects/NNcryoEM/data/rotamersdata/mrcs/MRCs_res30apix-1/2qim_res30apix13.mrc'
marker_file_name = 'C:/Mark/temp/Projects/NNcryoEM/data/temp/marker1.txt'
v = volume_list()[0]

points = [(10.243, 12.534, 25.352),
          (54.2112, 32.352, 29.3525),
          (-15.153, 139.23, 12.44)]
values = v.interpolated_values(points)
print values

def load_marker_file_name(file_name):
	with open(file_name) as f:
		lines = f.readlines()
	# skip first line
	mrks=[]
	for line in lines[1:]:
		wrds = line.strip().split(',')
		mrks.append({'x':float(wrds[0]),'y':float(wrds[1]),'z':float(wrds[2]),'label':wrds[3],'conf':wrds[4]})

	return mrks


#clear session
runCommand('close all')

#load map
whole_map = open_volume_file(map_name)[0]
mrks = load_marker_file_name(marker_file_name)

for mrk in mrks:
	command_str = 'shape sphere  color blue radius 1 coordinateSystem #0 center ' +str(mrk['x']) +','+str(mrk['y']) +','+str(mrk['z'])
	runCommand(command_str)
