from chimera import runCommand
from MarkChimeraUtils import atomsList2spec
import VolumeViewer


VOX_SIZE = 2.0
RESOLUTION = 6.0
elementsName = ['C', 'S', 'O', 'N']

def read_list_file(list_file):
    emdcodes=[]
    pdbcodes =[]
    with open(list_file) as fp:
        for cnt, line in enumerate(fp):
            words = line.split()
            emdcodes.append(words[0])
            pdbcodes.append(words[-1])

    return zip((emdcodes, pdbcodes))



def calc_voxalization_by_atom_type(pdb_id,grid3D,res=RESOLUTION):

    Id_for_copy = 4001
    Id_for_molmap = 5001

    vx_map = {}

    ##loop on atoms
    for at_name in elementsName:
        ##copy structure
        runCommand('combine #{} name  {}_atoms modelId {}'.\
                   format(pdb_id, at_name,Id_for_copy))
        ## delete atoms
        runCommand('delete #{}:@/element!={}'.format(Id_for_copy,at_name))
        ## run molmap
        runCommand('molmap #{} {}  modelId {}'.\
                   format(Id_for_copy,res,Id_for_molmap))


        #extract matrix (copy?)
        vx_map[at_name]=map_to_matrix(Id_for_molmap,grid3D)

        # delete copied structure
        runCommand('close #{}'.format(Id_for_copy))
        # delete mol map
        runCommand('close #{}'.format(Id_for_molmap))

    return vx_map

def get_object_by_id(id):
    all_objs = filter(lambda x:x.id==id, chimera.openModels.list())
    return all_objs[0]

def calc_3D_grid(pdb_id,vx_size,res):

    syth_map_id = 6001
    #molmap
    runCommand('molmap #{} {} gridSpacing {} modelId {}'\
               .format(pdb_id,res,vx_size,syth_map_id))

    #extract grid
    v_obj = get_object_by_id(syth_map_id)

    Xmin,Xmax = v_obj.xyz_bounds()
    xr = np.arange(Xmin[0]+vx_size/3,Xmax[0],vx_size)
    yr = np.arange(Xmin[1]+vx_size/3,Xmax[1],vx_size)
    zr = np.arange(Xmin[2]+vx_size/3,Xmax[2],vx_size)
    Xs,Ys,Zs = np.meshgrid(xr,yr,zr,indexing='ij')

    #remove models
    runCommand('close #{}'.format(syth_map_id))
    #return output
    return Xs,Ys,Zs

def map_to_matrix(map_id,grid3D):
    v_obj = get_object_by_id(map_id)
    Xs,Ys,Zs = grid3D
    xyz_coor = np.vstack((np.reshape(Xs,-1),np.reshape(Ys,-1),np.reshape(Zs,-1))).transpose()
    values, outside = v_obj.interpolated_values(xyz_coor,out_of_bounds_list = True)
    mtrx = np.reshape(values,Xs.shape)

    return mtrx

def calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION):
    prot1 = chimera.openModels.open(pdb_file)[0]
    map_obj = VolumeViewer.volume.open_volume_file(map_file)[0]
    pdb_id = prot1.id
    map_id = map_obj.id

    Xs,Ys,Zs = calc_3D_grid(pdb_id,vx_size,res)

    output_mtrx = map_to_matrix(map_id,(Xs,Ys,Zs))
    local_fit_matrix=[]
    inp_mtrc = calc_voxalization_by_atom_type(pdb_id,(Xs,Ys,Zs),res=res)

    return inp_mtrc, output_mtrx, local_fit_matrix

def save_matrc_to_debug(inp_mtrc,output_mtrx):
    np.save('output',output_mtrx)
    for at_name in elementsName:
        np.save(at_name,inp_mtrc[at_name])
    return




runCommand('close all')
#pdb_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/6j2c.cif"
#map_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/emd-2984.map"
pdb_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/test1.pdb"
map_file = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/t1.mrc"
inp_mtrc, output_mtrx, local_fit_matrix = calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION)
