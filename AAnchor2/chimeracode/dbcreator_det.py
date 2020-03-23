import os
import sys
import numpy as np
import copy
from glob import glob, glob1
import cPickle
import gzip



if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
utils_path = dir_path + '/../pythoncode/utils/'
sys.path.append(utils_path)


from kdtree import KDTree4
from process_rotamers_data import read_rotamers_data_text_file
from process_rotamers_data import get_pdb_id
import dbloader
from dbloader import LabelbyAAType, save_6tuple_data,save_det_labeled_5tuple_data


defoult_rotamers_file_name = dir_path +'/../data/rotamersdata/DatasetForBBDepRL2010.txt'
DEFAULT_DIST_THRESHOLD = 1.5
DEFAULT_NONES_RATIO = 0.3
TEMP_FOLDER =  dir_path +'/../data/temp/'
DEFAULT_STEP_FOR_DETECTION = 2
DEBUG_FILE = TEMP_FOLDER+'debug.txt'

class NoNormalization(object):
    @staticmethod
    def normamlize_3D_box( box):
        return box

class Mean0Sig1Normalization(object):
    @staticmethod
    def normamlize_3D_box(bx):
        bx_norm = (bx-np.mean(bx))/np.sqrt(np.var(bx))
        return bx_norm

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
    def save_map_positions_as_pklgz(input_map,output_map):
        #load
        whole_map = open_volume_file(input_map,model_id = 17)[0]
        #get_full matrix
        data_mtrx = whole_map.full_matrix()
        #calc_positions
        Nx,Ny,Nz = data_mtrx.shape
        x_pos = np.zeros((Nx,Ny,Nz))
        y_pos = np.zeros((Nx,Ny,Nz))
        z_pos = np.zeros((Nx,Ny,Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    x,y,z = whole_map.ijk_to_global_xyz((i,j,k))
                    x_pos[i,j,k] = x
                    y_pos[i,j,k] = y
                    z_pos[i,j,k] = z
        f = gzip.open(output_map, "w")
        cPickle.dump((data_mtrx, x_pos,y_pos,z_pos), f)
        f.close()
        return

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



# class SlidindWindowDB(object):
#     def __init__(self, apix=1.0,label = LabelbyAAType, box_center = BoxCenterAtCG,normalization = NoNormalization,  cubic_box_size = 11, distance_threshold =1,step = 2, dist_thr = 2, dilute_ratio = 1.0/500):
#
#         self.apix = apix
#         self.box_size = cubic_box_size
#         self.normalization = normalization
#         self.step = step
#         self.dist_thr = dist_thr
#         self.box_center = box_center
#         self.label = label
#         self.dilute_ratio = dilute_ratio
#
#
#
#
#
#     def calc_all_unlabeled_boxes(self, mrc_file):
#         #clear chimera
#         #load map
#         init_map = open_volume_file(mrc_file,model_id=17)[0]
#         #resample map
#         resampled_map = self.resample_map_with_chimera(init_map)
#         # extract boxes and box centers
#         Nx,Ny,Nz = resampled_map.full_matrix().shape
#         boxes = []
#         centers = []
#         map_mean = mean = np.mean(resampled_map.full_matrix())
#         map_std = np.sqrt(np.var(resampled_map.full_matrix()))
#
#         for x in range(0,Nx-self.box_size,self.step):
#             for y in range(0,Ny-self.box_size,self.step):
#                 for z in range(0,Nz-self.box_size,self.step):
#                     box_no_norm = resampled_map.full_matrix()[x:x+self.box_size,y:y+self.box_size,z:z+self.box_size]
#                     if np.mean(box_no_norm)<(map_mean-0*map_std):
#                         continue
#                     box_norm = self.normalization.normamlize_3D_box(box_no_norm)
#                     boxes.append(box_norm)
#                     x_c = x+self.box_size/2
#                     y_c = y+self.box_size/2
#                     z_c = y+self.box_size/2
#                     centers.append(resampled_map.ijk_to_global_xyz((x_c,y_c,z_c)))
#
#         runCommand('close all')
#         return  boxes, centers
#
#
#     def create_and_save_unlabeled_data(self, mrc_file,filename = dir_path + '/../data/temp/def_boxes.pkl.gz' ):
#         boxes, centers = self.calc_all_unlabeled_boxes(mrc_file)
#         self.save_unlabeled_data(boxes, centers,filename)
#         return
#     def save_labeled_data(self, boxes, centers,labels,filename = dir_path + '/../data/temp/def_boxes.pkl.gz' ):
#         f = gzip.open(filename, "w")
#         cPickle.dump((boxes, centers,labels), f)
#         f.close()
#
#
#     def create_and_save_labeled_data(self, mrc_file,pdb_file,filename = dir_path + '/../data/temp/def_boxes.pkl.gz' ):
#         runCommand('close all')
#         boxes, centers,labels = self.calc_all_labeled_boxes(mrc_file,pdb_file)
#         self.save_labeled_data(boxes, centers,labels,filename)
#         return
#
#
# class ClassificationFromRotamers(object):
#     def __init__(self, input_pdb_folder, mrc_maps_folder,target_folder,file_name_suffix,list_file_name=defoult_rotamers_file_name,apix=1.0,label = LabelbyAAType,box_center = BoxCenterAtCG,
#     normalization = NoNormalization,     cubic_box_size = 11):
#         """         """
#         self.input_pdb_folder = input_pdb_folder+'/'
#         self.mrc_maps_folder = mrc_maps_folder+'/'
#         self.target_folder = target_folder+'/'
#         self.file_name_suffix = file_name_suffix
#         self.apix = apix
#         self.rotamers_by_pdb_dict = read_rotamers_data_text_file(list_file_name)
#         self.label = label
#         self.label_statistics = {}
#         self.calc_all_labels()
#         self.box_center = box_center
#         self.normalization = normalization
#         self.cubic_box_size = cubic_box_size
#
#     def print_labels_to_file(self,file_name):
#         self.label.print_labels(file_name)
#
#
#     def calc_all_labels(self):
#         for pdb_id in self.rotamers_by_pdb_dict.keys():
#             dict_of_chains = self.rotamers_by_pdb_dict[pdb_id]
#             for chain_ID in dict_of_chains.keys():
#                 dict_of_resnums = dict_of_chains[chain_ID]
#                 for resnum in dict_of_resnums.keys():
#                     res_data = dict_of_resnums[resnum]
#                     label_num = self.label.calc_label(res_data)
#                     res_data["label"] = self.label.calc_label(res_data)
#                     self.label_statistics[label_num] = self.label_statistics.get(label_num,0)+1
#         return
#
#     def save_labeled_pairs(self, training_pairs, debug_data=[],filename="NONE"):
#     	if filename== "NONE":
#     		filename = self.target_folder+"default.pkl.gz"
#     	f = gzip.open(filename, "w")
#     	cPickle.dump((training_pairs, debug_data), f)
#     	f.close()
#
#     def test_residue_data(self,res_struct, res_data):
#         struct_type = 	res_struct.type
#         list_type = res_data["Type"]
#         if struct_type == list_type:
#             return True
#         if struct_type == 'PRO' and list_type in ["TPR","CPR"]:
#             return True
#         if struct_type == 'CYS' and list_type in ["CYH","CYD"]:
#     	    return True
#         return False
#
#
#     def calc_all_labeled_pairs_for_pdb(self, pdb_id):
#         #mrc file name
#         mrc_file_names = glob(self.mrc_maps_folder + pdb_id +'*.mrc')
#         mrc_file_name = mrc_file_names[0]
#         #assert len(mrc_file_names00)==1
#
#         box_centers,labels = self.calc_box_centers_and_labels(pdb_id)
#         boxes_xyz = [box_from_box_center(bx[0],bx[1],bx[2],self.apix,self.cubic_box_size) for bx in box_centers]
#         boxes_data = EMmaps.extract_3D_boxes(mrc_file_name,boxes_xyz,self.normalization)
#
#         labeled_pairs = (boxes_data,labels)
#         debug_data = [] #for future debuggi
#
#         return labeled_pairs, debug_data
#
#     def calc_and_save_all_labeled_pairs_for_pdb(self, pdb_id,file_to_save):
#         if file_to_save ==[]:
#             file_to_save= self.target_folder + pdb_id +'_'+ self.file_name_suffix
#         labeled_pairs,  debug_data=self.calc_all_labeled_pairs_for_pdb(pdb_id)
#         self.save_labeled_pairs(labeled_pairs, debug_data,filename = file_to_save)
#         return


class DBcreator(object):
    def __init__(self, input_pdb_folder = TEMP_FOLDER, mrc_maps_folder = TEMP_FOLDER,target_folder = TEMP_FOLDER, file_name_prefix = 'DBfrom_', list_file_name=defoult_rotamers_file_name, apix=1.0, label = LabelbyAAType, box_center = BoxCenterAtCG,
    normalization = NoNormalization, cubic_box_size = 11,dist_thr = DEFAULT_DIST_THRESHOLD, nones_ratio = DEFAULT_NONES_RATIO,step_for_detection = DEFAULT_STEP_FOR_DETECTION,debug_file=DEBUG_FILE,rand_position = False):
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
        self.nones_ratio = nones_ratio
        self.step = step_for_detection
        self.debug_file = debug_file
        self.rand_position = rand_position
        f= open(debug_file,'w')
        f.close()

    def write_to_debug_file(self,line_to_write):
        f= open(self.debug_file,'a')
        f.write(line_to_write+'\n')
        f.close()



    def get_residues_from_all_models(self,pdb_file):
        all_mdls = chimera.openModels.open(pdb_file,'PDB')
        res_list = []
        for mdl in all_mdls:
            res_list = res_list + mdl.residues
        return res_list

    def calc_box_centers_and_labels_from_pdb(self,pdb_file_name):

        box_centers =[]
        labels = []
        all_res_list = self.get_residues_from_all_models(pdb_file_name)
        centers = []
        labels = []
        for res_strct in all_res_list:
            x,y,z = self.box_center.get_box_center(res_strct)
            centers.append( np.asarray([x,y,z]))
            labels.append(self.label.calc_label_from(res_strct.type))
        return centers,labels

    def calc_nones(self,centers,n):
        centers_arr = np.asarray(centers)
        x_min = np.amin(centers_arr[:,0])
        x_max = np.amax(centers_arr[:,0])
        y_min = np.amin(centers_arr[:,1])
        y_max = np.amax(centers_arr[:,1])
        z_min = np.amin(centers_arr[:,2])
        z_max = np.amax(centers_arr[:,2])

        x_pos = np.random.uniform(x_min,x_max,n)
        y_pos = np.random.uniform(y_min,y_max,n)
        z_pos = np.random.uniform(z_min,z_max,n)

        kdt = KDTree4(centers)

        none_centers =[]
        for ind in range(n):
            center = np.asarray([x_pos[ind],y_pos[ind],z_pos[ind]])
            if len(kdt.in_range(center,self.dist_thr)) == 0:
                none_centers.append(center)
        return none_centers

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

    def create_det_db_labeled(self,mrc_file,pdb_file,file_name = [] ):

        pdb_file_full_name = self.input_pdb_folder+pdb_file
        mrc_file_full_name = self.mrc_maps_folder+mrc_file

        if file_name==[]:
            file_name = self.file_name_prefix + mrc_file[:-4]+self.file_name_suffix
        file_to_save = self.target_folder+file_name

        labels_pdb, kdt_centers = self.calc_labels_kdtree(pdb_file_full_name)
        #clear chimera
        runCommand('close all')
        #load map
        init_map = open_volume_file(mrc_file_full_name,model_id=17)[0]
        # calc positions and labels
        data_mtrx = init_map.full_matrix()
        #calc_positions
        Nx,Ny,Nz = data_mtrx.shape
        x_pos = np.zeros((Nx,Ny,Nz))
        y_pos = np.zeros((Nx,Ny,Nz))
        z_pos = np.zeros((Nx,Ny,Nz))
        labels = np.zeros((Nx,Ny,Nz))
        kk=0
        for i in range(Nx):
            if np.sum(data_mtrx[i,:,:]) ==0.0:
                print "DEBUG i",i
                continue
            for j in range(Ny):
                if np.sum(data_mtrx[:,j,:]) ==0.0:
                    print "DEBUG j",j
                    continue
                for k in range(Nz):
                    if np.sum(data_mtrx[:,:,k]) ==0.0:
                        print "DEBUG k",k
                        continue

                    x,y,z = init_map.ijk_to_global_xyz((i,j,k))
                    x_pos[i,j,k] = x
                    y_pos[i,j,k] = y
                    z_pos[i,j,k] = z
                    in_range = kdt_centers.in_range((x_pos[i,j,k],y_pos[i,j,k],z_pos[i,j,k]), self.dist_thr)
                    assert len(in_range)<=1
                    if len(in_range)==1:
                        labels[i,j,k] = labels_pdb[in_range[0]]
                    kk+=1
                    if kk%1000 ==0:
                        print "DEBUG",kk,Nx*Ny*Nz

        save_det_labeled_5tuple_data(file_to_save,data_mtrx,x_pos,y_pos,z_pos,labels)
        runCommand('close all')
        return

    def randomize_centers(self,centers):
        c1 = np.asarray(centers)
        d = np.random.uniform(low=0,high=self.dist_thr/np.sqrt(3),size = c1.shape)
        c2 = c1+d
        centers =[c2[k] for k in range(c1.shape[0])]
        return centers

    def create_classification_db(self,mrc_file,pdb_file,file_name = [],file_name_suffix = '.pkl.gz' ):


        self.write_to_debug_file(mrc_file + '###' +pdb_file)

        pdb_file_full_name = self.input_pdb_folder+pdb_file
        mrc_file_full_name = self.mrc_maps_folder+mrc_file

        if file_name==[]:
            file_name = self.file_name_prefix + mrc_file[:-4]+file_name_suffix
        file_to_save = self.target_folder+file_name

        #calc all boxes centers
        centers,labels = self.calc_box_centers_and_labels_from_pdb(pdb_file_full_name)
        if self.rand_position:
            centers = self.randomize_centers(centers)

        ## add nones
        n_labeled = len(centers)
        n_nones = int(round(n_labeled*self.nones_ratio))
        none_centers = self.calc_nones(centers,n_nones)
        none_labels = [self.label.calc_label_from('NONE')]*len(none_centers)

        centers_all = centers+none_centers
        labels_all = labels+none_labels

        boxes_xyz = [self.box_from_box_center(bx[0],bx[1],bx[2]) for bx in centers_all]
        boxes_all = EMmaps.extract_3D_boxes(mrc_file_full_name,boxes_xyz,self.normalization,inst_for_debug=self)
        orientations_all = [(0,0,0)]*len(labels_all)
        mrc_files = [mrc_file]*len(labels_all)
        pdb_files = [pdb_file]*len(labels_all)


        ind_filtered = list(filter(lambda n: boxes_all[n][0][0][0] != -999, range(len(boxes_all))))
        boxes_all_filtered = [boxes_all[x] for x in ind_filtered]
        centers_all_filtered = [centers_all[x] for x in ind_filtered]
        labels_all_filtered = [labels_all[x] for x in ind_filtered]
        orientations_all_filtered = [orientations_all[x] for x in ind_filtered]
        mrc_files_filtered = [mrc_files[x] for x in ind_filtered]
        pdb_files_filtered = [pdb_files[x] for x in ind_filtered]

        save_6tuple_data(file_to_save,boxes_all_filtered,centers_all_filtered,labels_all_filtered,orientations_all_filtered,mrc_files_filtered,pdb_files_filtered)

        return

    def resample_map_with_chimera(self, initial_map):

        def get_non_zero_grid(input_map):
            all_bounds = input_map.xyz_bounds()
            full_map = input_map.full_matrix()

            x_min =  0
            x_max =  full_map.shape[0]
            while np.sum(full_map[x_min+1,:,:]) ==0.0:
                x_min +=1
            while np.sum(full_map[x_max-2,:,:]) ==0.0:
                x_max -=1

            y_min =  0
            y_max =  full_map.shape[1]
            while np.sum(full_map[:,y_min+1,:]) ==0:
                y_min +=1
            while np.sum(full_map[:,y_max-2,:])==0:
                y_max -=1

            z_min =  0
            z_max =  full_map.shape[2]
            while np.sum(full_map[:,:,z_min+1]) ==0:
                z_min +=1
            while np.sum(full_map[:,:,z_max-2]) ==0:
                z_max -=1
            x_min_A,y_min_A,z_min_A = input_map.ijk_to_global_xyz((x_min,y_min,z_min))
            x_max_A,y_max_A,z_max_A = input_map.ijk_to_global_xyz((x_max,y_max,z_max))
            print
            return x_min_A,x_max_A,y_min_A,y_max_A,z_min_A,z_max_A

        assert initial_map.id == 17

        x_min,x_max,y_min,y_max,z_min,z_max = get_non_zero_grid(initial_map)

        x_grid = np.arange(x_min-self.apix,x_max+self.apix,self.apix)
        y_grid = np.arange(y_min-self.apix,y_max+self.apix,self.apix)
        z_grid =  np.arange(z_min-self.apix,z_max+self.apix,self.apix)

        x, y, z = np.meshgrid(x_grid, x_grid, x_grid)

        origin_xyz = (x_min,y_min,z_min)

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
