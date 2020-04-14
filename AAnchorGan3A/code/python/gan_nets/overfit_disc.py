

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
base_data_folder = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/"
data_fld = base_data_folder + "/res6/synth_exp/"
out_fld = base_data_folder + "/results/disc_exp/"


model_path = out_fld+'/network_test/'
graph_folder = out_fld+'/graphs/'
test_res_folder = out_fld + '/tests/'

if os.path.isdir(out_fld):
    shutil.rmtree(out_fld, ignore_errors=True)

os.mkdir(out_fld)
os.mkdir(model_path)
os.mkdir(graph_folder)
os.mkdir(test_res_folder)



syn_data_pairs = dataset_loader.read_list_file(data_fld+'list_synth.txt')
syn_pdbs = [x[0] for x in  syn_data_pairs]
real_data_pairs = dataset_loader.read_list_file(data_fld+'list_real.txt')
real_pdbs = [x[0] for x in  real_data_pairs]

all_data = EM_DATA_REAL_SYTH(data_fld,real_pdbs = real_pdbs[:4],synth_pdbs =syn_pdbs[:4], is_random = False)


# In[3]:


importlib.reload(net_3d_1)
#tf.reset_default_graph()
nn = net_3d_1.DISC_V1()
all_iter = all_data.train_dataset.make_initializable_iterator()
all_pair = all_iter.get_next()


# In[4]:


# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    time_start = timer()
    sess.run(all_iter.initializer)
    # training-loop

    for batch in range(all_data.N_batches-1):
            maps, labels = sess.run(all_pair)

            p_real = all_data.train_points[55900]
            p_synth = all_data.train_points[1655900]

            map_real = dataset_loader.get_out_map(p_real)
            map_synth = dataset_loader.get_out_map(p_synth)

            label_r = dataset_loader.get_real_synth(p_real)
            label_s = dataset_loader.get_real_synth(p_synth)



            for i in range(dataset_loader.BATCH_SIZE):
                if np.random.random()>0.5:
                    maps[i,:,:,:,:] = 10
                    labels[i,:,:,:,:] = 1#label_r
                else:
                    maps[i,:,:,:,:] = map_synth
                    labels[i,:,:,:,:] =  0 #label_s

            feed_dict={nn.x: maps, nn.x_label: labels, nn.keep_prob: 0.8, nn.isTrain: True}

            loss_d, acc_d, _= sess.run([nn.disc_loss["loss"], nn.disc_loss["acc"],nn.D_optim],                                       feed_dict={nn.x: maps, nn.x_label: labels, nn.keep_prob: 0.8, nn.isTrain: True})

            if batch %100 == 0:
            #sess.run(nn.clip)
                print("D Loss  {} Disc Acc {} ".format(loss_d,acc_d))


# In[ ]:


p_real = all_data.train_points[55900]
p_synth = all_data.train_points[1655900]

map_real = dataset_loader.get_out_map(p_real)
map_synth = dataset_loader.get_out_map(p_synth)

label_r = dataset_loader.get_real_synth(p_real)
label_s = dataset_loader.get_real_synth(p_synth)


# In[ ]:


print(nn.isTrain)


# In[ ]:


n, bins, patches = plt.hist(x=np.reshape(map_patch,-1),bins=10,range=[-2.0,2])


# In[ ]:


print(np.std(map_patch),np.mean(map_patch))


# In[ ]:





# In[ ]:
