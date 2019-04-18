import os.path
import numpy as np

try:
  import tensorflow as tf
except ImportError:
  print("run without TENSORFLOW")


VX_FILE_SUFF={'C':'_C','S':'_S','H':'_H','N':'_N','O':'_O'}
VOX_SIZE = 2.0
RESOLUTION = 6.0
NBOX_IN = 9
NBOX_OUT = 5
N_SAMPLS_FOR_1V3 = 1.0/(5.0**3)
N_CHANNELS = 5

BATCH_SIZE = 256


class TestBox():
    def __init__(self, p):
        feature = np.concatenate((getbox(p.vx_dict['C'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                    getbox(p.vx_dict['O'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                    getbox(p.vx_dict['N'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                    getbox(p.vx_dict['S'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                    getbox(p.vx_dict['H'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN)),\
                    axis =3)
        label = getbox(p.out,p.ijk[0],p.ijk[1],p.ijk[2],NBOX_OUT)

        self.pdb_id  = pdb_id
        self.coords = p.ijk
        self.inputs = feature
        self.out = None
        self.target = label
        return

def getbox(mp,I,J,K,NN):

    return mp[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1,np.newaxis]

class MolData():
    def __init__(self,f_names,pdb_id):
        self.out = np.load(f_names['OUT'])
        self.vx = {}
        self.vx['C'] = np.load(f_names['C'])
        self.vx['N'] = np.load(f_names['N'])
        self.vx['O'] = np.load(f_names['O'])
        self.vx['S'] = np.load(f_names['S'])
        self.vx['H'] = np.load(f_names['H'])

        self.lcc = np.load(f_names['LCC'])
        self.pdb_id = pdb_id

class DataPoint():
    def __init__(self,x,y,z,map,vx_dict,pdb_id,is_real = True):
        self.ijk = (x,y,z)
        self.out = map
        self.vx_dict = vx_dict
        self.pdb_id = pdb_id
        self.is_real = is_real

class EM_DATA_DISC_RANDOM():

    def __init__(self,folder_name, train_pdbs = []):
        #load data
        self.mols_train = load_pdb_list(folder_name, train_pdbs)

        #create points
        self.train_points = points_from_mols(self.mols_train.values())
        #ranomize train points
        self.train_points =np.random.permutation(self.train_points)


        self.N_train = len(self.train_points)

        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data_random(self.train_points)

        self.feature_shape = [NBOX_OUT  ,NBOX_OUT, NBOX_OUT,1]
        self.label_shape = [1,1,1,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape))).\
                        batch(BATCH_SIZE).shuffle(buffer_size=100)
        return



def points_from_mols(mols, is_real = True):
    points=[]
    for ml in mols:
        all_coords = np.nonzero(ml.lcc)
        for i,j in enumerate(all_coords[0]):
            points.append(DataPoint(all_coords[0][i],all_coords[1][i],all_coords[2][i]\
            , ml.out, ml.vx, ml.pdb_id,is_real = is_real))
    return points

def generator_from_data_random(points_data):
    def gen():
        for p in points_data:
            label = np.random.choice([0,1])*np.ones([1,1,1,1])
            if label[0][0][0][0] >0:
                map_patch = getbox(p.out,p.ijk[0],p.ijk[1],p.ijk[2],NBOX_OUT)
            else:
                map_patch = np.random.randn(NBOX_OUT,NBOX_OUT,NBOX_OUT,1)

            std = np.std(map_patch)
            avg = np.mean(map_patch)
            map_patch = (map_patch-avg)/std

            yield (map_patch,label)
    return gen


def generator_from_data_real_synth(points_data):
    def gen():
        for p in points_data:
            if p.is_real:
                label = np.ones([1,1,1,1])
            else:
                label = np.zeros([1,1,1,1])
            map_patch = getbox(p.out,p.ijk[0],p.ijk[1],p.ijk[2],NBOX_OUT)
            std = np.std(map_patch)
            avg = np.mean(map_patch)
            map_patch = (map_patch-avg)/std

            yield (map_patch,label)
    return gen

class EM_DATA_REAL_SYTH():

    def __init__(self,folder_name, real_pdbs = [],synth_pdbs =[], is_random = True):
        #load data
        self.mols_real = load_pdb_list(folder_name, real_pdbs)
        self.mols_synth = load_pdb_list(folder_name, synth_pdbs)

        #create points
        self.real_points = points_from_mols(self.mols_real.values(), is_real = True)
        self.synth_points = points_from_mols(self.mols_synth.values(), is_real = False)
        self.train_points = self.real_points + self.synth_points

        #ranomize train points
        if is_random:
            self.train_points =np.random.permutation(self.train_points)


        self.N_train = len(self.train_points)

        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data_real_synth(self.train_points)

        self.feature_shape = [NBOX_OUT,NBOX_OUT,NBOX_OUT,1]
        self.label_shape = [1,1,1,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape)))
        self.train_dataset = self.train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=100)
        return


class EM_DATA():

    def __init__(self,folder_name, train_pdbs = [],is_random = True):
        #load data
        self.mols_train = load_pdb_list(folder_name, train_pdbs)

        #create points
        self.train_points = points_from_mols(self.mols_train.values())
        #ranomize train points
        if is_random:
            self.train_points =np.random.permutation(self.train_points)


        self.N_train = len(self.train_points)

        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data(self.train_points)

        self.feature_shape = [NBOX_IN  ,NBOX_IN, NBOX_IN,N_CHANNELS]
        self.label_shape = [NBOX_OUT,NBOX_OUT,NBOX_OUT,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape))).\
                        batch(BATCH_SIZE).shuffle(buffer_size=100)
        return

def load_pdb_list(folder_name, pdb_list):
    mols={}
    for pdb_id in pdb_list:
        try:
            pdb_files = get_and_check_file_names(pdb_id,folder_name)
            mol_data = MolData(pdb_files,pdb_id)
            mols[pdb_id] = mol_data
            print("{} Loaded".format(pdb_id))
        except Exception as e:
            print("{} FAILED, Error : ".format(pdb_id, str(e)))
    return mols


def generator_from_data(points_data):
    def gen():
        for p in points_data:
            feature = np.concatenate((getbox(p.vx_dict['C'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['O'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['N'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['S'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['H'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN)),\
                        axis =3)
            label = getbox(p.out,p.ijk[0],p.ijk[1],p.ijk[2],NBOX_OUT)
            std = np.std(label)
            avg = np.mean(label)
            label = (label  -avg)/std

            yield (feature,label)
    return gen

def generator_from_data_test(points_data):
    def gen():
        for num_point,p in enumerate(points_data):
            feature = np.concatenate((getbox(p.vx_dict['C'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['O'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['N'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['S'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN),\
                        getbox(p.vx_dict['H'],p.ijk[0],p.ijk[1],p.ijk[2],NBOX_IN)),\
                        axis =3)
            label = getbox(p.out,p.ijk[0],p.ijk[1],p.ijk[2],NBOX_OUT)
            print("DEBUG 235")
            yield (feature,label)
    return gen



def get_file_names(pdb_id,folder):
    f_names={}
    f_names['OUT'] = folder+'/'+'F_'+pdb_id+'_output.npy'
    for at in VX_FILE_SUFF.keys():
        f_names[at] = folder+'/'+'F_'+pdb_id+VX_FILE_SUFF[at]+'.npy'
    f_names['LCC'] = folder+'/'+'F_'+pdb_id+'_lcc.npy'
    return f_names



def get_and_check_file_names(pdb_id,folder):
    f_names = get_file_names(pdb_id,folder)
    for f_name in f_names.values():
        if not os.path.isfile(f_name):
            print(f_name)
            return {}
    return f_names

def read_list_file(list_file):
    pairs=[]
    with open(list_file) as fp:
        line = fp.readline()#read header
        line = fp.readline()
        while line:
            pdb_id = line[0:4]
            emd_id = line[5:9]
            res = float(line[10:13])
            line = fp.readline()
            pairs.append((pdb_id,emd_id,res))
    return pairs


# fld = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/temp_data"
# fn = get_and_check_file_names('pppp',fld)
#
#
#
# sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])
#
# def gg():
#     n=0
#     for el in sequence:
#         n+=1
#         yield(n, np.asarray(el))
#
# gen1 = gg
#
# #for i in range(2):
# #    a = next(gen1)
# #    print(a)
#
#
#
# dt = tf.data.Dataset.from_generator(gen1,(tf.int64,tf.int64),\
#                                            (tf.TensorShape([]),tf.TensorShape([None,1])))
#
# it1= dt.make_initializable_iterator()
# el = it1.get_next()
#
#
if __name__ == "__main__":
    fld = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/single_pdbs/"
    em1 = EM_DATA(fld,train_pdbs=['hhhh','nnnn'],test_pdbs=['oooo','cccc','ssss'])
    train_data = em1.train_dataset.make_initializable_iterator()
    test_data = em1.test_dataset.make_initializable_iterator()

    trn = train_data.get_next()
    tst = test_data.get_next()
    with tf.Session() as sess:
        sess.run(train_data.initializer)
        sess.run(test_data.initializer)
        for k in range(10):
            print('TRAIN')
            x,y = sess.run(trn)
            print(np.sign(x))
            print('TEST')
            print(sess.run(tst))
#     print(2)
#     print(sess.run(el))
#     print(3)
#     print(sess.run(el))
#     sess.run(it1.initializer)
#     print(4)
#     print(sess.run(el))
#     print(5)
#     print(sess.run(el))
#     print(6)
#     print(sess.run(el))
