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
data_fld = base_data_folder + "/res6/synth/"
out_fld = base_data_folder + "/results/disc/"

model_path = out_fld+'/network_test/'
graph_folder = out_fld+'/graphs/'
test_res_folder = out_fld + '/tests/'


disc_data_file = '4800.ckpt'

#check dirs
for d in [model_path,graph_folder,test_res_folder,data_fld]:
    if not os.path.isdir(d):
        print("FOLDER NOT EXISTS: " + d)


data_pairs = dataset_loader.read_list_file(data_fld+'list.txt')
all_pdbs = [x[0] for x in  data_pairs]



em_train = EM_DATA(data_fld,train_pdbs=all_pdbs[30:35], is_random = False)
em_data = em_train.train_dataset.make_initializable_iterator()
em_pair = em_data.get_next()


nn = net_3d_1.DISC_V1()

# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()

    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    sess.run(em_data.initializer)

    saver.restore(sess, model_path+disc_data_file)

    for batch in range(em_train.N_batches-1):
        features,maps = sess.run(em_pair)

        prediction_true = sess.run(nn.disc["o"],feed_dict={nn.x: maps, nn.isTrain: False})

        x_false = np.random.randn(BATCH_SIZE, NBOX_OUT,NBOX_OUT,NBOX_OUT,1)
        prediction_false = sess.run(nn.disc["o"],feed_dict={nn.x: x_false, nn.isTrain: False})


        print(np.count_nonzero(np.squeeze(prediction_true)>0.5),np.count_nonzero(np.squeeze(prediction_false)>0.5))
