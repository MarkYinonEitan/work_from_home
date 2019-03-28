import os.path
import numpy as np
import tensorflow as tf


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
    def __init__(self,x,y,z,map,vx_dict):
        self.ijk = (x,y,z)
        self.out = map
        self.vx_dict = vx_dict


class EM_DATA():

    def __init__(self,folder_name, box_size = 8,train_pdbs = [], \
                valid_ratio=0.2, test_pdbs=[] ):
        #load data
        self.mols_train = self.load_pdb_list(folder_name, train_pdbs)
        self.mols_test = self.load_pdb_list(folder_name, test_pdbs)

        #create points
        self.train_points = self.points_from_mols(self.mols_train.values())
        self.test_points = self.points_from_mols(self.mols_test.values())

        self.train_generator = generator_from_data(self.train_points)

        self.train_dataset = tf.data.Dataset.from_generator(self.train_generator,\
                        (tf.float64,tf.float64),(tf.TensorShape([None]),tf.TensorShape([None])))
        return

    def gen():
      #for i in itertools.count(1):
        for i in range(10):
            yield (i, np.asarray([1] * i))



    def load_pdb_list(self,folder_name, pdb_list):
        mols={}
        for pdb_id in pdb_list:
            pdb_files = get_and_check_file_names(pdb_id,folder_name)
            mol_data = MolData(pdb_files,pdb_id)
            mols[pdb_id] = mol_data
        return mols

    def points_from_mols(self,mols):
        points=[]
        for ml in mols:
            all_coords = np.nonzero(ml.lcc)
            for i,j in enumerate(all_coords[0]):
                points.append(DataPoint(all_coords[0][i],all_coords[1][i],all_coords[2][i]\
                ,ml.out,ml.vx))
        return points



def generator_from_data(points_data):
    def gen():
        for p in points_data:
            feature = [p.vx_dict['C'][p.ijk[0],p.ijk[1],p.ijk[2]],\
                        p.vx_dict['O'][p.ijk[0],p.ijk[1],p.ijk[2]],\
                        p.vx_dict['N'][p.ijk[0],p.ijk[1],p.ijk[2]],\
                        p.vx_dict['S'][p.ijk[0],p.ijk[1],p.ijk[2]],\
                        p.vx_dict['H'][p.ijk[0],p.ijk[1],p.ijk[2]]]
            label = p.out[p.ijk[0],p.ijk[1],p.ijk[2]]
            yield (feature,label)
    return gen


def get_and_check_file_names(pdb_id,folder):
    f_names={}
    f_names['OUT'] = folder+'/'+'F_'+pdb_id+'_output.npy'
    f_names['C'] = folder+'/'+'F_'+pdb_id+'_C.npy'
    f_names['N'] = folder+'/'+'F_'+pdb_id+'_N.npy'
    f_names['O'] = folder+'/'+'F_'+pdb_id+'_O.npy'
    f_names['S'] = folder+'/'+'F_'+pdb_id+'_S.npy'
    f_names['H'] = folder+'/'+'F_'+pdb_id+'_H.npy'
    f_names['LCC'] = folder+'/'+'F_'+pdb_id+'_lcc.npy'


    for f_name in f_names.values():
        if not os.path.isfile(f_name):
            print(f_name)
            return {}
    return f_names


fld = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/first_tests/temp_data"
fn = get_and_check_file_names('pppp',fld)



sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])

def gg():
    n=0
    for el in sequence:
        n+=1
        yield(n, np.asarray(el))

gen1 = gg

#for i in range(2):
#    a = next(gen1)
#    print(a)



dt = tf.data.Dataset.from_generator(gen1,(tf.int64,tf.int64),\
                                           (tf.TensorShape([]),tf.TensorShape([None,1])))

it1= dt.make_initializable_iterator()
el = it1.get_next()

with tf.Session() as sess:
    sess.run(it1.initializer)
    print(1)
    print(sess.run(el))
    print(2)
    print(sess.run(el))
    print(3)
    print(sess.run(el))
    sess.run(it1.initializer)
    print(4)
    print(sess.run(el))
    print(5)
    print(sess.run(el))
    print(6)
    print(sess.run(el))

em1 = EM_DATA(fld,box_size= 8,train_pdbs=['pppp','pppp'],test_pdbs=['pppp','pppp'])
