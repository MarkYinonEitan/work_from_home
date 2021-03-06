import chimera
import numpy as np
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import active_volume
from numpy import zeros, array, dot, linalg
from chimera import runCommand

def map_to_matrix(map_id,grid3D):
    v_obj = get_object_by_id(map_id)
    Xs,Ys,Zs = grid3D
    xyz_coor = np.vstack((np.reshape(Xs,-1),np.reshape(Ys,-1),np.reshape(Zs,-1))).transpose()
    values, outside = v_obj.interpolated_values(xyz_coor,out_of_bounds_list = True)
    mtrx = np.reshape(values,Xs.shape)

    return mtrx

def get_residues_from_all_models(pdb_file):
    all_mdls = chimera.openModels.open(pdb_file,'PDB')
    res_list = []
    for mdl in all_mdls:
        res_list = res_list + mdl.residues
    return res_list

def calc_voxalization_by_atom_type(pdb_id,grid3D,res=3.0 , atomTypes = []):

    Id_for_copy = 4001
    Id_for_molmap = 5001

    vx_map = {}

    ##loop on atoms
    for at_name in atomTypes:

        ##copy structure
        runCommand('combine #{} name  {}_atoms modelId {}'.\
                   format(pdb_id, at_name,Id_for_copy))
        ## delete atoms
        runCommand('delete #{}:@/element!={}'.format(Id_for_copy,at_name))
        ## run molmap
        no_atoms = get_object_by_id(Id_for_copy)==-1


        if no_atoms:
            runCommand('vop new zero_map origin {},{},{} modelId {}'.\
                       format(np.mean(grid3D[0]),np.mean(grid3D[1]),np.mean(grid3D[1]),Id_for_molmap))
        else:
            runCommand('molmap #{} {}  modelId {}'.\
                   format(Id_for_copy,res,Id_for_molmap))

        #extract matrix (copy?)
        vx_map[at_name]=map_to_matrix(Id_for_molmap,grid3D)

        # delete copied structure
        runCommand('close #{}'.format(Id_for_copy))
        # delete mol map
        runCommand('close #{}'.format(Id_for_molmap))

    return vx_map



def calc_3D_grid(pdb_id,vx_size,margin):

    syth_map_id = 6001
    #molmap
    res = vx_size*3.0
    runCommand('molmap #{} {} gridSpacing {} modelId {} replace false '\
               .format(pdb_id,res,vx_size,syth_map_id))
    #extract grid
    v_obj = get_object_by_id(syth_map_id)

    Xmin,Xmax = v_obj.xyz_bounds()
    xr = np.arange(Xmin[0]-margin,Xmax[0]+margin,vx_size)
    yr = np.arange(Xmin[1]-margin,Xmax[1]+margin,vx_size)
    zr = np.arange(Xmin[2]-margin,Xmax[2]+margin,vx_size)
    Xs,Ys,Zs = np.meshgrid(xr,yr,zr,indexing='ij')

    #remove models
    runCommand('close #{}'.format(syth_map_id))
    #return output
    return Xs,Ys,Zs


def get_rotamer_angles(res):
    angle_names = ["phi", "psi", "chi1", "chi2", "chi3", "chi4"]
    angles = np.array([res.phi, res.psi, res.chi1, res.chi2, res.chi3, res.chi4]).astype(float)
    angles[np.isnan(angles)] = -999
    ang_dict = {}
    for k in range(len(angle_names)):
        ang_dict[angle_names[k]] = angles[k]

    return ang_dict

def get_object_by_id(id):
    all_objs = filter(lambda x:x.id==id, chimera.openModels.list())
    if len(all_objs)==1:
        return all_objs[0]
    else:
        return -1

def getRotMatr2Res(res1,res2):
    x1,y1,z1 = getCB_CA_N_frame(res1)
    Rot1 = np.matrix([[x1[0], y1[0],z1[0]],
                         [x1[1], y1[1],z1[1]],
                         [x1[2], y1[2],z1[2]]])

    x2,y2,z2 = getCB_CA_N_frame(res2)
    Rot2 = np.matrix([[x2[0], y2[0],z2[0]],
                         [x2[1], y2[1],z2[1]],
                         [x2[2], y2[2],z2[2]]])



    Rot = np.transpose(Rot2)*Rot1
    return Rot, np.transpose(Rot1),np.transpose(Rot2)

def getCB_CA_N_frame(res1):

    #first res
    CA_atom = res1.findAtom('CA')
    CB_atom = res1.findAtom('CB')
    N_atom = res1.findAtom('N')

    x_axes = CB_atom.coord() - CA_atom.coord()
    y1_axes = N_atom.coord() - CA_atom.coord()


    z_axes = np.cross(x_axes,y1_axes)
    y_axes = -np.cross(x_axes,z_axes)
    x_axes = array(x_axes)
    x_axes = x_axes/np.linalg.norm(x_axes)
    y_axes = y_axes/np.linalg.norm(y_axes)
    z_axes = z_axes/np.linalg.norm(z_axes)

    #Rot = [x_axes'; y_axes';z_axes']
    return x_axes,y_axes,z_axes


def getResByNumInChain(prot, chainName, Num):
    for res in prot.residues:
        if res.id.chainId == chainName and res.id.position == Num:
            return res
    return []

def atomsList2spec(atomsList):
    spec = ""
    for atom in atomsList:
        mod = str(atom.molecule.id);
        sub_mod = str(atom.molecule.subid);
        chainId = atom.residue.id.chainId;
        resNum = str(atom.residue.id.position);
        atomType = atom.name
        altLoc = atom.altLoc

        spec = spec + ' ' +'#' +mod +'.' + sub_mod + ':' + resNum + '.' + chainId + '@' + atomType + '.' + altLoc

    return spec



def molmapCube(model_id, resolution):
    """
    creates map from model as the original mol map, but with cube grid
    """

    ####create grid and save as model
    #get gabarities
    min_x, max_x, min_y, max_y, min_z, max_z = getGabarities(model_id)
    #create empty cube map
    min_cor = min(min_x,min_y,min_z)
    max_cor = max(max_x,max_y,max_z)
    d_grid = resolution/3



    #run molmap
    molmap_com = 'molmap #'+str(model_id) + ' ' + str(resolution)+' gridSpacing ' + str(resolution/3.0)
    chimera.runCommand(molmap_com)
    map_orig = active_volume();

    # interpolation
    createCubeMapfromGivenMap(map_orig,min_cor, max_cor, d_grid)

    #delete the grid
    map_orig.destroy()



def getGabarities(model_id):
    #get list of atoms
    prot = chimera.openModels.list()[model_id]

    atom = prot.atoms[0]
    min_x = atom.coord()[0]
    max_x = atom.coord()[0]
    min_y = atom.coord()[1]
    max_y = atom.coord()[1]
    min_z = atom.coord()[2]
    max_z = atom.coord()[2]

    for atom in prot.atoms:
        x = atom.coord()[0]
        y = atom.coord()[1]
        z = atom.coord()[2]

        min_x = min(min_x,x)
        min_y = min(min_y,y)
        min_z = min(min_z,z)


        max_x = max(max_x,x)
        max_y = max(max_y,y)
        max_z = max(max_z,z)

    return min_x, max_x, min_y, max_y, min_z, max_z

def createCubeMapfromGivenMap(ref_map,min_cor, max_cor, d_grid):
    g= []
    for z_coor in drange(min_cor,max_cor,d_grid):
        y_list = [];
        for y_coor in drange(min_cor,max_cor,d_grid):
            x_list = []
            for x_coor in drange(min_cor,max_cor,d_grid):
                point_coordinate = (x_coor,y_coor,z_coor)
                x_list.append(point_coordinate);
            y_list.append(x_list);
        g.append(y_list)

    ga_3d = np.array(g);

    ga_shape = ga_3d.shape;
    ga_1d = np.reshape(ga_3d,[ga_shape[0]*ga_shape[1]*ga_shape[2], ga_shape[3]])

    # create original model
    map_region_model_1d =  ref_map.interpolated_values(ga_1d)
    map_region_model= np.reshape(map_region_model_1d,ga_shape[0:3])
    grid = Array_Grid_Data(map_region_model, (min_cor,min_cor,min_cor), (d_grid,d_grid,d_grid), name = 'EMRegion')
    v_orig = volume_from_grid_data(grid)

    return v_orig.id



    #calc SVD
    #plot

def drange(start, stop, step):
    r = start
    while r < stop:
     	yield r
     	r += step
