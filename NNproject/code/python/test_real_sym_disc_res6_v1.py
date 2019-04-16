import os
import sys
import importlib
import numpy as np
import shutil
import tensorflow as tf
from timeit import default_timer as timer
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path + '/'
sys.path.append(python_path)



import dataset_loader
importlib.reload(dataset_loader)
from dataset_loader import EM_DATA
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
import net_3d_1
import utils
from utils import get_available_gpus

N_EPOCHS = 30

DEVICE_GPU = get_available_gpus()

#define folders
base_data_folder = "/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/"
syn_fld = base_data_folder + "/res6/synth/"
real_fld = base_data_folder +"/res6/exp/"

out_fld = base_data_folder + "/results/disc_exp/"
model_path = out_fld+'/network_test/'


disc_data_file = '5400.ckpt'


syn_data_pairs = dataset_loader.read_list_file(syn_fld+'list.txt')
all_pdbs = [x[0] for x in  syn_data_pairs]
syn_data = EM_DATA(syn_fld,train_pdbs=all_pdbs[30:35], is_random = True)
syn_iter = syn_data.train_dataset.make_initializable_iterator()
syn_pair = syn_iter.get_next()

real_data_pairs = dataset_loader.read_list_file(real_fld+'list.txt')
all_pdbs = [x[0] for x in  real_data_pairs]
real_data = EM_DATA(real_fld,train_pdbs=all_pdbs[5:8], is_random = True)
real_iter = real_data.train_dataset.make_initializable_iterator()
real_pair = real_iter.get_next()


nn = net_3d_1.DISC_V1()

# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    sess.run(syn_iter.initializer)
    sess.run(real_iter.initializer)

    saver.restore(sess, model_path+disc_data_file)

    for batch in range(100):
        #synthetic
        features,maps_s = sess.run(syn_pair)
        prediction_syn = sess.run(nn.disc["o"],feed_dict={nn.x: maps_s, nn.isTrain: False})
        #random
        maps_rand = np.random.randn(BATCH_SIZE, NBOX_OUT,NBOX_OUT,NBOX_OUT,1)
        prediction_rand = sess.run(nn.disc["o"],feed_dict={nn.x: maps_rand, nn.isTrain: False})
        #real
        features,maps_r = sess.run(real_pair)
        prediction_real = sess.run(nn.disc["o"],feed_dict={nn.x: maps_r, nn.isTrain: False})

        print("SYNTH  RAND REAL")
        print(np.count_nonzero(np.squeeze(prediction_syn)>0.5),np.count_nonzero(np.squeeze(prediction_rand)>0.5),np.count_nonzero(np.squeeze(prediction_real)>0.5))
