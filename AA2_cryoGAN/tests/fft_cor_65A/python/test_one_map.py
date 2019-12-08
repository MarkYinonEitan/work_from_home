import os
import sys
import importlib
import numpy as np
import shutil
import tensorflow as tf
from timeit import default_timer as timer



python_path = '/Users/markroza/Documents/GitHub/work_from_home/AA2_cryoGAN/code/python'
sys.path.append(python_path)



import dataset_loader
importlib.reload(dataset_loader)
from dataset_loader import EM_DATA
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
import net_3d_5
import utils



#base_data_folder = "/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/"
base_data_folder = "/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/tests/fft_cor_65A/"
inp_data_fld = base_data_folder + "/vox6n18/"
out_fld = base_data_folder + "/gan_6n18/"
nets_fld = '/Users/markroza/Documents/work_from_home/data/AA2_cryoGAN/nets/'
#model_path = out_fld+'/network_test/'

vae_gan_file = nets_fld + "498000.ckpt"
disc_weights_file = nets_fld + "7100.ckpt"
disc_net = net_3d_5.DISC_V1()


pdb_id = '6nt8'

em_data = EM_DATA(inp_data_fld,train_pdbs=[pdb_id], is_random = False)

test_points_inp = utils.getTestPoints(em_data)
test_points_disc = utils.disc_test_points(test_points_inp, disc_net, disc_weights_file)


# open session and initialize all variables
config = tf.ConfigProto()

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver.restore(sess, vae_gan_file)



    kn={};
    for k in range(em_data.N_train):

        if k%BATCH_SIZE == 0:
            print(k, em_data.N_train)
            inp_batch = np.zeros([BATCH_SIZE,NBOX_IN,NBOX_IN,NBOX_IN,N_CHANNELS])
            out_batch = np.zeros([BATCH_SIZE,NBOX_OUT,NBOX_OUT,NBOX_OUT,1])
            n = 0

        p = em_data.train_points[k]
        inp_batch[n,:,:,:,:] = dataset_loader.get_inp_map(p)
        out_batch[n,:,:,:,:] = dataset_loader.get_out_map(p)

        kn[n]=k
        n=n+1

#        map_out = dataset_loader.get_out_map(p)
#        test_points[k].out = map_out

        if n == BATCH_SIZE and True:
            map_out, r_loss = sess.run([vae_gan.mp_fake, vae_gan.reconstr_loss],feed_dict={vae_gan.vx_real: inp_batch, vae_gan.mp_real: out_batch})
            print(r_loss, np.mean(out_batch),np.mean(map_out),np.std(out_batch),np.std(map_out))
            for i in range(BATCH_SIZE-1):
                test_points[kn[i]].out = test_points[kn[i]].out*0



disc_net = net_3d_5.DISC_V1()
run_disc_on_map(inp_map_file, out_map_file, disc_net, disc_wights_file)

disc_res = disc_test_points(test_points,disc_file)

out_matrx = tstpoint2mtrx(test_points)

out_file_name = out_fld +'test_'+pdb_id +'.npy'

np.save(out_file_name,out_matrx)
