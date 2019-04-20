

import os
import sys
import importlib
import numpy as np
import shutil
import pickle
import tensorflow as tf
from timeit import default_timer as timer
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path + '/../python/'
sys.path.append(python_path)



import dataset_loader
importlib.reload(dataset_loader)
from dataset_loader import EM_DATA_REAL_SYTH, EM_DATA,EM_DATA_DISC_RANDOM
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
from dataset_loader import getbox
import net_3d_1
import utils
from utils import get_available_gpus


# In[2]:


#define folders
#base_data_folder = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/"
base_data_folder = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/"
base_data_folder = "/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/"
data_fld = base_data_folder + "/res6/exp/"
out_fld = base_data_folder + "/results/vae_res6/"


model_path = out_fld+'/network_test/'
graph_folder = out_fld+'/graphs/'
test_res_folder = out_fld + '/tests/'

if os.path.isdir(out_fld):
    shutil.rmtree(out_fld, ignore_errors=True)

os.mkdir(out_fld)
os.mkdir(model_path)
os.mkdir(graph_folder)
os.mkdir(test_res_folder)



real_data_pairs = dataset_loader.read_list_file(data_fld+'list.txt')
real_pdbs = [x[0] for x in  real_data_pairs]


real_data = EM_DATA(data_fld,train_pdbs = real_pdbs[:5], is_random = True)
real_iter = real_data.train_dataset.make_initializable_iterator()
real_pair = real_iter.get_next()

real_data_t = EM_DATA(data_fld,train_pdbs = real_pdbs[5:], is_random = True)
real_iter_t = real_data_t.train_dataset.make_initializable_iterator()
real_pair_t = real_iter_t.get_next()


importlib.reload(net_3d_1)
#tf.reset_default_graph()
nn = net_3d_1.VAE_V1()


# In[4]:


# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    time_start = timer()
    sess.run(real_iter.initializer)
    sess.run(real_iter_t.initializer)
    # training-loop

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(graph_folder, sess.graph)

    n=0
    for batch in range(1000000):
            if n > real_data.N_batches-2:
                sess.run(real_iter.initializer)
                sess.run(real_iter_t.initializer)
                n=0
            n=n+1


            vx_real ,mp_real= sess.run(real_pair)
            fd ={nn.vx_real: vx_real, nn.mp_real: mp_real}

            sess.run(nn.opti_VAE, feed_dict=fd)

            #update
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            sess.run(extra_update_ops)

            #Learrate
            new_learn_rate = sess.run(nn.new_learning_rate)
            if new_learn_rate > 0.00005:
                sess.run(nn.add_global)


            if batch %10 == 0:
                loss_vae= sess.run(nn.generator_loss,feed_dict=fd)
                print("GEN  Loss   {} B {} N {}  ".format(loss_vae,batch,real_data.N_batches))
                #Summary
                summary_str = sess.run(summary_op, feed_dict=fd)
                summary_writer.add_summary(summary_str, batch)
            if batch %100 == 0:
                fd ={nn.vx_real: vx_real, nn.mp_real: mp_real}
                loss_vae= sess.run(nn.generator_loss,feed_dict=fd)
                print("Losses ",np.mean(loss_vae))
                saved_path = saver.save(sess, model_path + str(batch) + ".ckpt")
                print("Model Saved")
                #print("NEW LEARN RATE", new_learn_rate)
                map_gen = sess.run(nn.mp_fake,feed_dict=fd)
                res = {"inp":vx_real,"mp_real":mp_real,"mp_gen":map_gen}
                with  open(test_res_folder + str(batch) + ".pkl", 'wb') as out_file:
                    pickle.dump(res, out_file)
