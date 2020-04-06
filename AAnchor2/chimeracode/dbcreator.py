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

utils_path = cur_pass + '/../pythoncode/utils/'
chimera_path = cur_pass + '/../chimeracode/'
sys.path.append(utils_path)
sys.path.append(chimera_path)


from kdtree import KDTree4
from process_rotamers_data import read_rotamers_data_text_file
from process_rotamers_data import get_pdb_id
import dbloader
from dbloader import LabelbyAAType
from dbloader import Mean0Sig1Normalization, NoNormalization, global_xyz_to_ijk, get_boxes

data_folder  = '/specific/netapp5_2/iscb/wolfson/Mark/data/AAnchor/'
defoult_rotamers_file_name = data_folder  + '/rotamersdata/DatasetForBBDepRL2010.txt'
DEFAULT_DIST_THRESHOLD = 1.5
TEMP_FOLDER =  data_folder  + '/temp/'
DEFAULT_STEP_FOR_DETECTION = 2
DEBUG_FILE = 'debug.txt'


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
    def get_transformation_ijk_to_xyz(init_map):
        Nx,Ny,Nz = init_map.full_matrix().shape

        #calc transformation
        K=100
        i0 = np.random.randint(Nx,size=K)
        j0 = np.random.randint(Ny,size=K)
        k0 = np.random.randint(Nz,size=K)
        A=[]
        B=[]
        for (i,j,k) in zip(i0,j0,k0):
            x,y,z =  init_map.ijk_to_global_xyz((i,j,k))
            row1 = [i,j,k,1,0,0,0,0,0,0,0,0.0]
            b1 = x
            row2 = [0,0,0,0,i,j,k,1,0,0,0.0,0]
            b2 = y
            row3 = [0,0,0,0,0,0,0,0.0,i,j,k,1]
            b3 = z
            A.append(row1)
            A.append(row2)
            A.append(row3)
            B.append(b1)
            B.append(b2)
            B.append(b3)
        C = np.linalg.lstsq(np.asarray(A),np.asarray(B))[0]
        #test transformation
        i1 = np.random.randint(Nx,size=K)
        j1 = np.random.randint(Ny,size=K)
        k1 = np.random.randint(Nz,size=K)
        err = 0.0
        for (i,j,k) in zip(i1,j1,k1):
            x_m,y_m,z_m =  init_map.ijk_to_global_xyz((i,j,k))
            x = C[0]*i+C[1]*j+C[2]*k+C[3]
            y = C[4]*i+C[5]*j+C[6]*k+C[7]
            z = C[8]*i+C[9]*j+C[10]*k +C[11]
            err+=(x_m-x)**2+(y_m-y)**2+(z_m-z)**2

        assert err <00.0001,"Error is {}".format(err)
        return C


    @staticmethod
    def extract_3D_boxes(map_file,boxes_xyz,normalization,inst_for_debug = 'NONE'):
        """
        boxes_xyz - list of boxs_xyz
        box_xyz - tuple (X,Y,Z)
        X - 3D np array of x coordinates
        Y - 3D np array of y coordinates
        Z - 3D np array of z coordinates
        X.shape = Y.shape = Z.shape
        """

        def get_one_box(v,box_xyz):
            assert box_xyz[0].shape == box_xyz[1].shape
            assert box_xyz[1].shape == box_xyz[2].shape

            points =  zip(np.reshape(box_xyz[0],-1),np.reshape(box_xyz[1],-1),np.reshape(box_xyz[2],-1))
            values = v.interpolated_values(points)
            box = np.reshape(values, box_xyz[1].shape)

            return box

        #load map
        runCommand('close all')
        whole_map = open_volume_file(map_file,model_id = 17)[0]

        if inst_for_debug !='NONE':
            inst_for_debug.write_to_debug_file(str(whole_map.full_matrix().shape))


        #check that it is only map
        v = volume_list()[0]
        assert v.id == 17
        assert v == whole_map
        map_mean = mean = np.mean(v.full_matrix())
        map_std = np.sqrt(np.var(v.full_matrix()))
        boxes_data = []
        for bx in boxes_xyz:
            box_no_norm = get_one_box(v,bx)
            if np.mean(box_no_norm)<(map_mean-0*map_std):
                boxes_data.append(-999*np.ones(box_no_norm.shape))
            else:
                boxes_data.append(normalization.normamlize_3D_box(box_no_norm))

        return boxes_data

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
    def __init__(self, input_pdb_folder = TEMP_FOLDER, mrc_maps_folder = TEMP_FOLDER,target_folder = TEMP_FOLDER, file_name_prefix = 'DBfrom_', list_file_name=defoult_rotamers_file_name, apix=1.0, label = LabelbyAAType, box_center = BoxCenterAtCG,
    normalization = NoNormalization, cubic_box_size = 11,dist_thr = DEFAULT_DIST_THRESHOLD, step_for_detection = DEFAULT_STEP_FOR_DETECTION,debug_file=DEBUG_FILE,is_corners = False,        use_list = False ):
        """         """
        self.input_pdb_folder = input_pdb_folder+'/'
        self.mrc_maps_folder = mrc_maps_folder+'/'
        self.target_folder = target_folder+'/'
        self.file_name_prefix = file_name_prefix
        self.apix = apix
        self.rotamers_by_pdb_dict = read_rotamers_data_text_file(list_file_name)
        self.label = label
        self.label_statistics = {}
        self.box_center = box_center
        self.normalization = normalization
        self.cubic_box_size = cubic_box_size
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

    def get_residues_from_all_models(self,pdb_file):
        all_mdls = chimera.openModels.open(self.input_pdb_folder+pdb_file,'PDB')
        res_list = []
        for mdl in all_mdls:
            res_list = res_list + mdl.residues
        return res_list

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
        all_res_list = self.get_residues_from_all_models(pdb_file_name)
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

        print "DEBUG FILTER BOX 0 ", limits_pdb
        print "DEBUG FILTER BOX", len(all_res_list),len(centers),len(labels)
        return centers, labels

    
    def box_from_box_center(self,x,y,z):

        cbs = self.cubic_box_size
        apx = self.apix
        x_gr = np.linspace(x-cbs/2*apx,x+cbs/2*apx,cbs)
        y_gr = np.linspace(y-cbs/2*apx,y+cbs/2*apx,cbs)
        z_gr = np.linspace(z-cbs/2*apx,z+cbs/2*apx,cbs)
        X,Y,Z = np.meshgrid(x_gr,y_gr,z_gr,indexing = 'ij')
        box_xyz = (X,Y,Z)

        return box_xyz

    def calc_labels_kdtree(self,pdb_file):
        #extract aa and centers
        ## loop on all residues in this pdb
        all_res_list = self.get_residues_from_all_models(pdb_file)
        centers = []
        labels = []
        for res_strct in all_res_list:
            x,y,z = self.box_center.get_box_center(res_strct)
            centers.append( np.asarray([x,y,z]))
            labels.append(self.label.calc_label_from(res_strct.type))
        #create kd tree
        kdt = KDTree4(centers)

        return labels , kdt


    def centers_to_corners(self,centers,labels):
        assert self.apix ==1.0
        c0 = np.asarray(centers)

        c_ddd = np.vstack((np.floor(c0[:,0]),np.floor(c0[:,1]),np.floor(c0[:,2]))).T
        c_ddu = np.vstack((np.floor(c0[:,0]),np.floor(c0[:,1]),np.ceil(c0[:,2]))).T
        c_dud = np.vstack((np.floor(c0[:,0]),np.ceil(c0[:,1]),np.floor(c0[:,2]))).T
        c_duu = np.vstack((np.floor(c0[:,0]),np.ceil(c0[:,1]),np.ceil(c0[:,2]))).T
        c_udd = np.vstack((np.ceil(c0[:,0]),np.floor(c0[:,1]),np.floor(c0[:,2]))).T
        c_udu = np.vstack((np.ceil(c0[:,0]),np.floor(c0[:,1]),np.ceil(c0[:,2]))).T
        c_uud = np.vstack((np.ceil(c0[:,0]),np.ceil(c0[:,1]),np.floor(c0[:,2]))).T
        c_uuu = np.vstack((np.ceil(c0[:,0]),np.ceil(c0[:,1]),np.ceil(c0[:,2]))).T
        c_corners = np.concatenate((c_ddd,c_ddu,c_dud,c_duu,c_udd,c_udu,c_uud,c_uuu))


        labels_corners = []



        for k in range (8):
            for in_data in range(len(labels)):
                labels_corners.append(copy.deepcopy(labels[in_data]))

        c_corners_list = [c_corners[k,:] for k in range(c_corners.shape[0])]
        #correct positions

        for k in range(len(c_corners_list)):
            labels_corners[k]["box_center_x"] = c_corners_list[k][0]
            labels_corners[k]["box_center_y"] = c_corners_list[k][1]
            labels_corners[k]["box_center_z"] = c_corners_list[k][2]

        return c_corners_list,labels_corners

    def calc_limits_xyz(self, centers_list):
        cent_arr = np.asarray(centers_list)

        x_min = np.min(cent_arr[:,0])-self.cubic_box_size/2-5*self.apix
        y_min = np.min(cent_arr[:,1])-self.cubic_box_size/2-5*self.apix
        z_min = np.min(cent_arr[:,2])-self.cubic_box_size/2-5*self.apix

        x_max = np.max(cent_arr[:,0])+self.cubic_box_size/2+5*self.apix
        y_max = np.max(cent_arr[:,1])+self.cubic_box_size/2+5*self.apix
        z_max = np.max(cent_arr[:,2])+self.cubic_box_size/2+5*self.apix

        return ([x_min,x_max],[y_min,y_max],[z_min,z_max])

    def create_class_db_corners(self,mrc_file,pdb_file,file_name = [],file_name_suffix = '',limits_pdb=([-10.0**6,10.0**6],[-10.0**6,10.0**6],[-10.0**6,10.0**6]) ):

        print("DEBUG 3435",mrc_file,limits_pdb )
        pdb_file_full_name = self.input_pdb_folder+pdb_file
        mrc_file_full_name = self.mrc_maps_folder+mrc_file

        print ("DEBUG 8832", file_name==[])
        if file_name==[]:
            file_name = self.file_name_prefix + mrc_file[:-4]+file_name_suffix
        file_name_pref = self.target_folder+file_name

        #calc all boxes centers
        centers_pdb,labels_pdb= self.calc_box_centers_and_labels_from_pdb(pdb_file,check_list = self.use_list,limits_pdb = limits_pdb)
        print( "DEBUG 44", len(labels_pdb) ,mrc_file,limits_pdb )
        if len(labels_pdb)==0:
            print("DEBUG NO RES' in the BOX",mrc_file,limits_pdb )
            return


        centers_corners,labels_corners =   self.centers_to_corners(centers_pdb,labels_pdb)

        limits_xyz = self.calc_limits_xyz(centers_corners)

        #clear chimera
        runCommand('close all')
        #load map
        init_map = open_volume_file(mrc_file_full_name,model_id=17)[0]
        resampled_map = self.resample_map_with_chimera(init_map,limits_in = limits_xyz,limits_out = limits_pdb)
        # calc positions and labels
        data_mtrx = resampled_map.full_matrix()
        #calc transformation
        C = EMmaps.get_transformation_ijk_to_xyz(resampled_map)
        #calc filter matrix
        filter_matrix = np.ones(data_mtrx.shape)
        #swap data
        mtrx_not_swapped = data_mtrx
        filter_matrix_not_swapped = filter_matrix
        data_mtrx = np.swapaxes(mtrx_not_swapped,0,2)
        filter_matrix = np.swapaxes(filter_matrix_not_swapped,0,2)

        box_centers_ijk = global_xyz_to_ijk(np.asarray(centers_corners),C)

        pred_boxes,_= get_boxes(data_mtrx,box_centers_ijk,C,Mean0Sig1Normalization)

        dbloader.save_label_data_to_csv(pred_boxes,labels_corners , file_name_pref)

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

        x_grid = np.arange(x_min-self.apix,x_max+self.apix,self.apix)
        y_grid = np.arange(y_min-self.apix,y_max+self.apix,self.apix)
        z_grid =  np.arange(z_min-self.apix,z_max+self.apix,self.apix)

        x, y, z = np.meshgrid(x_grid, x_grid, x_grid)

        origin_xyz = (x_min,y_min,z_min)

        #move origin to cornerd

        new_map_command = 'vop new  int_map  size  %d,%d,%d ' %(len(x_grid),len(y_grid),len(z_grid))
        new_map_command = new_map_command + ' gridSpacing %s ' %self.apix
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
