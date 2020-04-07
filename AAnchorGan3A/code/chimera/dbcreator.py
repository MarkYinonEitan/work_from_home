import os
import sys
import numpy as np
import copy
import chimera
from glob import glob, glob1
from chimera import runCommand
from VolumeData import Array_Grid_Data
from VolumeViewer import volume_from_grid_data
from VolumeViewer import open_volume_file
from VolumeViewer import volume_list
from Matrix import euler_xform
import pickle
import gzip
import MarkChimeraUtils
reload(MarkChimeraUtils)


cur_pass = os.path.dirname(os.path.realpath(__file__))

python_path = cur_pass + '/../python/'
sys.path.append(python_path)

from kdtree import KDTree4
from process_rotamers_data import read_rotamers_data_text_file
from process_rotamers_data import get_pdb_id
import dbloader
from dbloader import LabelbyAAType
from dbloader import Mean0Sig1Normalization, NoNormalization
from dbloader import VX_BOX_SIZE, MAP_BOX_SIZE, RESOLUTION, VOX_SIZE, ATOM_NAMES

data_folder  = '/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/'
defoult_rotamers_file_name = data_folder  + '/rotamersdata/DatasetForBBDepRL2010.txt'
DEFAULT_DIST_THRESHOLD = 1.5
TEMP_FOLDER =  data_folder  + '/temp/'
DEFAULT_STEP_FOR_DETECTION = 2
DEBUG_FILE = 'debug.txt'

def calc_all_matrices(pdb_file, map_file,vx_size = VOX_SIZE, res = RESOLUTION):
    prot1 = chimera.openModels.open(pdb_file)[0]
    map_obj = open_volume_file(map_file)[0]
    pdb_id = prot1.id
    map_id = map_obj.id

    #add hydrogens
    runCommand('addh spec #{}'.format(pdb_id))
    margin = VX_BOX_SIZE*VOX_SIZE
    Xs,Ys,Zs = MarkChimeraUtils.calc_3D_grid(pdb_id,vx_size,margin)
    grid3D = (Xs,Ys,Zs)
    em_mtrx = MarkChimeraUtils.map_to_matrix(map_id,grid3D)
    vx_mtrc = MarkChimeraUtils.calc_voxalization_by_atom_type(pdb_id,grid3D,res=res, atomTypes = ATOM_NAMES)

    return em_mtrx, vx_mtrc, grid3D


def get_regions(pdb_file,N):

    #get all atoms
    all_mdls = chimera.openModels.open(pdb_file,'PDB')
    atoms_list = []
    for mdl in all_mdls:
        atoms_list = atoms_list + mdl.atoms

    (x_min,x_max,y_min,y_max,z_min,z_max)=(10.0**6,-10.0**6,10.0**6,-10.0**6,10.0**6,-10.0**6)
    for at in atoms_list:
        x_min = min(at.coord()[0],x_min)
        x_max = max(at.coord()[0],x_max)
        y_min = min(at.coord()[1],y_min)
        y_max = max(at.coord()[1],y_max)
        z_min = min(at.coord()[2],z_min)
        z_max = max(at.coord()[2],z_max)

    x_bounds = np.linspace(x_min,x_max,N+1)
    y_bounds = np.linspace(y_min,y_max,N+1)
    z_bounds = np.linspace(z_min,z_max,N+1)

    box_limits =[]
    for inx in range(N):
        for iny in range(N):
            for inz in range(N):
                box_limits.append(((x_bounds[inx],x_bounds[inx+1]),(y_bounds[iny],y_bounds[iny+1]),(z_bounds[inz],z_bounds[inz+1])))

    return box_limits


class BoxCenterAtCG(object):
    @staticmethod
    def get_box_center(residue):
        cg =np.array([0,0,0])
        natoms = 0.0
        for atom in residue.atoms:
            cg = cg + atom.coord()
            natoms+=1
        cg = cg/natoms
        return cg[0],cg[1],cg[2]


class EMmaps(object):

    @staticmethod
    def rotate_map_and_pdb(source_map_file,source_pdb_file,euler_angles_tuple,target_map_file,target_pdb_file):
        #load
        runCommand('close all')
        prot = chimera.openModels.open(source_pdb_file)[0]
        init_map = open_volume_file(source_map_file,model_id=17)[0]
        #rotate
        euler_angles = list(euler_angles_tuple)
        translation = [0.0,0.0,0.0]
        xf = euler_xform(euler_angles, translation)
        grid_rotated = init_map.copy()
        grid_rotated.openState.localXform(xf)
        runCommand('vop resample #%s onGrid #%s boundingGrid true modelId 47' %(grid_rotated.id,init_map.id))
        mxf_prot = prot.openState.xform.inverse()
        for a in prot.atoms:
            a.setCoord(mxf_prot.apply(xf.apply(a.xformCoord())))

        map_rotated = chimera.openModels.list()[-1]
        assert map_rotated.id == 47
        runCommand('close #{}'.format(grid_rotated.id))
        runCommand('close #{}'.format(init_map.id))
        #save
        runCommand('volume #47 save ' + target_map_file)
        runCommand('write #{} {}'.format(prot.id, target_pdb_file))
        #remove
        runCommand('close all')

        return



class DBcreator(object):
    def __init__(self, input_pdb_folder = TEMP_FOLDER, mrc_maps_folder = TEMP_FOLDER,target_folder = TEMP_FOLDER, file_name_prefix = 'DBfrom_', list_file_name=defoult_rotamers_file_name,  label = LabelbyAAType, box_center = BoxCenterAtCG, resolution = 3.0,    normalization = NoNormalization, dist_thr = DEFAULT_DIST_THRESHOLD, step_for_detection = DEFAULT_STEP_FOR_DETECTION,debug_file=DEBUG_FILE,is_corners = False,        use_list = False ):
        """         """
        self.input_pdb_folder = input_pdb_folder+'/'
        self.mrc_maps_folder = mrc_maps_folder+'/'
        self.target_folder = target_folder+'/'
        self.file_name_prefix = file_name_prefix
        self.resolution = resolution
        self.rotamers_by_pdb_dict = read_rotamers_data_text_file(list_file_name)
        self.label = label
        self.label_statistics = {}
        self.box_center = box_center
        self.normalization = normalization
        self.dist_thr = dist_thr
        self.step = step_for_detection
        self.debug_file = debug_file
        self.is_corners = is_corners
        self.use_list = use_list
        f= open(debug_file,'w')
        f.close()

    def write_to_debug_file(self,line_to_write):
        f= open(self.debug_file,'a')
        f.write(line_to_write+'\n')
        f.close()



    def is_in_list(self, res_struct,pdb_id):
        if not (pdb_id in self.rotamers_by_pdb_dict.keys()):
            return False
        dict_of_chains = self.rotamers_by_pdb_dict[pdb_id]
        chain_ID = res_struct.id.chainId
        if not chain_ID in dict_of_chains.keys():
            return False
        resnum = res_struct.id.position
        resType = res_struct.type
        if ((resType == 'ALA') or (resType == 'GLY')):
            return True
        dict_of_resnums = dict_of_chains[chain_ID]

        if not resnum in dict_of_resnums.keys():
            return False
        if  self.label.calc_label(dict_of_resnums[resnum]) !=  self.label.calc_label_from(res_struct.type):
            print "DEBUG FILTER Different res num"
            return False
        return True

    def calc_box_centers_and_labels_from_pdb(self,pdb_file_name, check_list = False,limits_pdb=([-10.0**6,10.0**6],[-10.0**6,10.0**6],[-10.0**6,10.0**6])):


        pdb_id = get_pdb_id(pdb_file_name)

        box_centers =[]
        labels = []
        all_res_list = MarkChimeraUtils.get_residues_from_all_models(pdb_file_name)
        if check_list:
            all_res_list = list(filter(lambda x: self.is_in_list(x,pdb_id), all_res_list))

        centers = []
        labels = []
        ref_data = []
        rotamers_data = []
        for res_strct in all_res_list:
            lb =  self.label.calc_label_from(res_strct.type)
            if lb!=-1:
                x,y,z = self.box_center.get_box_center(res_strct)
                if ((x<limits_pdb[0][0]) or (x>limits_pdb[0][1])):
                    continue
                if ((y<limits_pdb[1][0]) or (y>limits_pdb[1][1])):
                    continue
                if ((z<limits_pdb[2][0]) or (z>limits_pdb[2][1])):
                    continue
                centers.append( np.asarray([x,y,z]))

                lb_row={}
                lb_row["label"] = lb
                lb_row["chainId"] = res_strct.id.chainId
                lb_row["pos"] = res_strct.id.position
                lb_row["pdb_id"] = pdb_id
                lb_row["CG_pos_X"] = x
                lb_row["CG_pos_Y"] = y
                lb_row["CG_pos_Z"] = z

                rotamers_data = MarkChimeraUtils.get_rotamer_angles(res_strct)

                for ky in list(rotamers_data.keys()):
                    lb_row[ky] = rotamers_data[ky]

                labels.append(lb_row)
        return centers, labels



    def calc_labels_kdtree(self,pdb_file):
        #extract aa and centers
        ## loop on all residues in this pdb
        all_res_list = MarkChimeraUtils.get_residues_from_all_models(pdb_file)
        centers = []
        labels = []
        for res_strct in all_res_list:
            x,y,z = self.box_center.get_box_center(res_strct)
            centers.append( np.asarray([x,y,z]))
            labels.append(self.label.calc_label_from(res_strct.type))
        #create kd tree
        kdt = KDTree4(centers)

        return labels , kdt


    def centers_to_corners(self,centers,labels, grid3D):
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])

        Xs,Ys,Zs = grid3D

        X_ax = Xs[:,0,0]
        Y_ax = Ys[0,:,0]
        Z_ax = Zs[0,0,:]
        assert is_sorted(X_ax) and is_sorted(Y_ax) and is_sorted(Z_ax)

        c0 = np.asarray(centers)

        in_x_max = np.searchsorted(X_ax,c0[:,0],'left')
        in_x_min = in_x_max-1
        xmax = X_ax[in_x_max]
        xmin = X_ax[in_x_min]
        in_y_max = np.searchsorted(Y_ax,c0[:,1],'left')
        in_y_min = in_y_max-1
        ymax = Y_ax[in_y_max]
        ymin = Y_ax[in_y_min]
        in_z_max = np.searchsorted(Z_ax,c0[:,2],'left')
        in_z_min = in_z_max-1
        zmax = Z_ax[in_z_max]
        zmin = Z_ax[in_z_min]

        c_ddd = np.vstack((xmin,ymin,zmin)).T
        c_ddu = np.vstack((xmin,ymin,zmax)).T
        c_dud = np.vstack((xmin,ymax,zmin)).T
        c_duu = np.vstack((xmin,ymax,zmax)).T
        c_udd = np.vstack((xmax,ymin,zmin)).T
        c_udu = np.vstack((xmax,ymin,zmax)).T
        c_uud = np.vstack((xmax,ymax,zmin)).T
        c_uuu = np.vstack((xmax,ymax,zmax)).T
        c_corners = np.concatenate((c_ddd,c_ddu,c_dud,c_duu,c_udd,c_udu,c_uud,c_uuu))

        ind_ddd = np.vstack((in_x_min,in_y_min,in_z_min)).T
        ind_ddu = np.vstack((in_x_min,in_y_min,in_z_max)).T
        ind_dud = np.vstack((in_x_min,in_y_max,in_z_min)).T
        ind_duu = np.vstack((in_x_min,in_y_max,in_z_max)).T
        ind_udd = np.vstack((in_x_max,in_y_min,in_z_min)).T
        ind_udu = np.vstack((in_x_max,in_y_min,in_z_max)).T
        ind_uud = np.vstack((in_x_max,in_y_max,in_z_min)).T
        ind_uuu = np.vstack((in_x_max,in_y_max,in_z_max)).T
        ind_corners = np.concatenate((ind_ddd,ind_ddu,ind_dud,ind_duu,ind_udd,ind_udu,ind_uud,ind_uuu))

        labels_corners = []

        for k in range (8):
            for in_data in range(len(labels)):
                labels_corners.append(copy.deepcopy(labels[in_data]))

        c_corners_list = [c_corners[k,:] for k in range(c_corners.shape[0])]
        ind_corners_list = [ind_corners[k,:] for k in range(ind_corners.shape[0])]
        #correct positions

        for k in range(len(c_corners_list)):
            labels_corners[k]["box_center_x"] = c_corners_list[k][0]
            labels_corners[k]["box_center_y"] = c_corners_list[k][1]
            labels_corners[k]["box_center_z"] = c_corners_list[k][2]

        return ind_corners_list,c_corners_list,labels_corners

    def calc_limits_xyz(self, centers_list):
        cent_arr = np.asarray(centers_list)

        box_size = max(VX_BOX_SIZE,MAP_BOX_SIZE)

        x_min = np.min(cent_arr[:,0])-box_size/2-5*VOX_SIZE
        y_min = np.min(cent_arr[:,1])-box_size/2-5*VOX_SIZE
        z_min = np.min(cent_arr[:,2])-box_size/2-5*VOX_SIZE

        x_max = np.max(cent_arr[:,0])+box_size/2+5*VOX_SIZE
        y_max = np.max(cent_arr[:,1])+box_size/2+5*VOX_SIZE
        z_max = np.max(cent_arr[:,2])+box_size/2+5*VOX_SIZE

        return ([x_min,x_max],[y_min,y_max],[z_min,z_max])

    def create_class_db_corners(self,mrc_file,pdb_file,file_name = [],file_name_suffix = '', map_source = 'UNKNOWN', limits_pdb=([-10.0**6,10.0**6],[-10.0**6,10.0**6],[-10.0**6,10.0**6]) ):

        pdb_file_full_name = self.input_pdb_folder+pdb_file
        mrc_file_full_name = self.mrc_maps_folder+mrc_file

        if file_name==[]:
            file_name = self.file_name_prefix + mrc_file[:-4]+file_name_suffix
        file_name_pref = self.target_folder+file_name


        #class calc 3D grid
        runCommand('close all')
        #calc all boxes centers
        centers_pdb,labels_pdb= self.calc_box_centers_and_labels_from_pdb(pdb_file_full_name,check_list = self.use_list,limits_pdb = limits_pdb)
        if len(labels_pdb)==0:
            print("DEBUG NO RES' in the BOX",mrc_file,limits_pdb )
            return

        #add source
        for l_data in labels_pdb:
            l_data["MAP_SOURCE"] = map_source

        em_mtrx, vx_mtrc, grid3D = calc_all_matrices(pdb_file_full_name, mrc_file_full_name,vx_size = VOX_SIZE, res = RESOLUTION)
        centers_indexes, centers_corners,labels_corners =   self.centers_to_corners(centers_pdb,labels_pdb,grid3D)

        vx_boxes = []
        pred_boxes = []
        for  ind_cent in centers_indexes:
            box4d = dbloader.getbox(em_mtrx, ind_cent[0], ind_cent[1], ind_cent[2], MAP_BOX_SIZE, normalization = self.normalization)

            pred_boxes.append(np.squeeze(box4d))
            voxaliztion_box={}
            for at_name in vx_mtrc.keys():
                voxaliztion_box[at_name] =  dbloader.getbox(vx_mtrc[at_name], ind_cent[0], ind_cent[1], ind_cent[2], VX_BOX_SIZE, normalization = dbloader.NoNormalization)
            vx_boxes.append(voxaliztion_box)


        dbloader.save_label_data_to_csv(pred_boxes, vx_boxes, labels_corners , file_name_pref)

        return


    def resample_map_with_chimera(self, initial_map,limits_in=([10.0**6,-10.0**6],[10.0**6,-10.0**6],[10.0**6,-10.0**6]),limits_out=([-10.0**6,10.0**6],[-10.0**6,+10.0**6],[-10.0**6,+10.0**6])):

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

        x_grid = np.arange(x_min-VOX_SIZE, x_max+VOX_SIZE, VOX_SIZE)
        y_grid = np.arange(y_min-VOX_SIZE, y_max+VOX_SIZE, VOX_SIZE)
        z_grid = np.arange(z_min-VOX_SIZE, z_max+VOX_SIZE, VOX_SIZE)

        x, y, z = np.meshgrid(x_grid, x_grid, x_grid)

        origin_xyz = (x_min,y_min,z_min)

        #move origin to cornerd

        new_map_command = 'vop new  int_map  size  %d,%d,%d ' %(len(x_grid),len(y_grid),len(z_grid))
        new_map_command = new_map_command + ' gridSpacing %s ' %VOX_SIZE
        new_map_command = new_map_command + 'origin  %s,%s,%s' %(origin_xyz[0],origin_xyz[1],origin_xyz[2])
        new_map_command = new_map_command + ' modelId  27'
        runCommand(new_map_command)

        resample_map_command =  'vop resample  #{}  onGrid #{} modelId 37'.format(initial_map.id, '27')
        runCommand(resample_map_command)
        resampled_map = chimera.openModels.list()[2]
        for mdl in chimera.openModels.list():
            print "DEBUG model if" , mdl.id
        assert resampled_map.id == 37
        runCommand('close 27')
        return resampled_map
