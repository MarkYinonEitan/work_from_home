import os.path
import numpy as np

try:
  import tensorflow as tf
except ImportError:
  print("run without TENSORFLOW")

import dbloader


ATOM_NAMES=["C","S","H","N","O"]
VOX_SIZE = 1.0
RESOLUTION = 3.0
VX_BOX_SIZE = 15
MAP_BOX_SIZE = 11
N_SAMPLS_FOR_1V3 = 1.0/(2.0**3)
N_CHANNELS = 5

MEAN = 0.5
SIGMA = MEAN/3.0

BATCH_SIZE = 256


def getbox(mp,I,J,K,NN, normalization = dbloader.NoNormalization, mean = MEAN, sigma = SIGMA):
    bx_no_norm = mp[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1,np.newaxis]
    bx_norm = normalization.normamlize_3D_box(bx_no_norm, mean = mean, sigma = sigma)
    return bx_norm

class EM_DATA_DISC_RANDOM():

    def __init__(self,folder_name, train_pdbs = []):
        #load data
        self.full_file_names = [folder_name+'/'+x for x in train_pdbs]
        self.train_data_dict = {}
        dbloader.load_train_data_to_dict(self.full_file_names, self.train_data_dict)

        self.N_train = len(self.train_data_dict["boxes"])


        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data_random(self.train_data_dict)

        self.feature_shape = [MAP_BOX_SIZE  ,MAP_BOX_SIZE, MAP_BOX_SIZE,1]
        self.label_shape = [1,1,1,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape))).\
                        batch(BATCH_SIZE).shuffle(buffer_size=100)
        return

def generator_from_data_random(data_dict):
    def gen():
        for in_box in range(data_dict["boxes"])
            label = np.random.choice([0,1])*np.ones([1,1,1,1])
            map_patch = data_dict["boxes"][in_box]

            if label[0][0][0][0] < 0:
                mean  = np.mean(map_patch)
                sigma = np.sqrt(np.var(map_patch))
                map_patch = np.random.standard_normal(map_patch.shape)*sigma+mean

            yield (map_patch,label)
    return gen

def permute_train_dict(train_data_dict):
    self.N_train = len(self.train_data_dict["boxes"])

    in_x = np.random.permutation(N_train)
    train_data_dict["boxes"] = [train_data_dict["boxes"][k] for k in in_x]
    train_data_dict["data"] = [train_data_dict["data"][k] for k in in_x]
    train_data_dict["vx"] = [train_data_dict["vx"][k] for k in in_x]
    return




class EM_DATA_REAL_SYTH():

    def __init__(self,folder_name, real_pdbs = [],synth_pdbs =[], is_random = True):
        #load data
        self.full_file_names_real = [folder_name+'/'+x for x in real_pdbs]
        self.train_data_dict_real = {}
        dbloader.load_train_data_to_dict(self.full_file_names_real, self.train_data_dict_real)

        self.full_file_names_synth = [folder_name+'/'+x for x in synth_pdbs]
        self.train_data_dict_synth = {}
        dbloader.load_train_data_to_dict(self.full_file_names_synth, self.train_data_dict_synth)

        #create points
        for ky in in list(self.full_file_names_synth):
            self.train_data_dict[ky] = self.train_data_dict_real[ky] + self.train_data_dict_synth[ky]


        #ranomize train points
        if is_random:
            permute_train_dict(self.train_data_dict)

        self.N_train = len(self.train_data_dict["boxes"])

        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data_real_synth(self.self.train_data_dict)

        self.feature_shape = [MAP_BOX_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE,1]
        self.label_shape = [1,1,1,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape)))
        self.train_dataset = self.train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=100)
        return


class EM_DATA():

    def __init__(self,folder_name, train_pdbs = [],is_random = True):
        #load data
        self.full_file_names = [folder_name+'/'+x for x in train_pdbs]
        self.train_data_dict = {}
        dbloader.load_train_data_to_dict(self.full_file_names, self.train_data_dict)

        self.N_train = len(self.train_data_dict["boxes"])

        #ranomize train points
        if is_random:
            permute_train_dict(train_data_dict)

        self.N_batches = self.N_train // BATCH_SIZE

        self.train_generator = generator_from_data(self.train_data_dict)

        self.feature_shape = [VX_BOX_SIZE  ,VX_BOX_SIZE, VX_BOX_SIZE,N_CHANNELS]
        self.label_shape = [MAP_BOX_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE,1]

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float32,tf.float32),(tf.TensorShape(self.feature_shape),tf.TensorShape(self.label_shape)))
        self.train_dataset = self.train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=100)

        return

def data_file_name( file_pref, folder_name=""):
            # load data - valid
    file_name_csv = folder_name + file_name_pref + ".csv"
    file_name_map = folder_namefile_name_pref + ".mp"
    file_name_vox = file_name_pref + ".vx"

    return file_name_csv, file_name_map, file_name_vox

def generator_from_data(data_dict):
    def gen():
        for in_box in range(data_dict["boxes"]):
            feature = data_dict["vx"][in_box]
            label = data_dict["boxes"][in_box]

            yield (feature,label)
    return gen

def generator_from_data_real_synth(dict_data):
    def gen():
        for in_box in range(data_dict["boxes"])
            label_data = dict_data["data"][in_box]
            if label_data["MAP_SOURCE"] = "REAL":
                label = np.ones([1,1,1,1])
            elif label_data["MAP_SOURCE"] = "UNKNOWN":
                raise NameError('UNKNOWN MAP SOURCE ' + label_data["pdb_id"])
            else :
                label = np.zeros([1,1,1,1])
            map_patch = data_dict["boxes"][in_box]

            yield (map_patch,label)
    return gen

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
