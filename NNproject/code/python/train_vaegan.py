

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
import net_3d_5
import utils
from utils import get_available_gpus


Init_files={}
Init_files["DISC"] = "/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/results/disc_exp/network_test/1300.ckpt"

Init_files["VAE"] ="/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/results/vae_res6/network_test/1200.ckpt"

#define folders
#base_data_folder = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/"
base_data_folder = "/Users/markroza/Documents/work_from_home/NNcourse_project/data/"
#base_data_folder = "/specific/netapp5_2/iscb/wolfson/Mark/data/NNcourse_project/data/"
data_fld = base_data_folder + "/res6/synth_exp/"
out_fld = base_data_folder + "/results/vaegan_res6/"


model_path = out_fld+'/network_test/'
graph_folder = out_fld+'/graphs/'
test_res_folder = out_fld + '/tests/'

if os.path.isdir(out_fld):
    shutil.rmtree(out_fld, ignore_errors=True)

os.mkdir(out_fld)
os.mkdir(model_path)
os.mkdir(graph_folder)
os.mkdir(test_res_folder)

real_train_pdbs = ['5fik']#,'5a31','5l9t','6n88']
real_test_pdbs = ['5fik']

real_data = EM_DATA(data_fld,train_pdbs = real_train_pdbs, is_random = True)
real_iter = real_data.train_dataset.make_initializable_iterator()
real_pair = real_iter.get_next()

real_data_t = EM_DATA(data_fld,train_pdbs = real_test_pdbs, is_random = True)
real_iter_t = real_data_t.train_dataset.make_initializable_iterator()
real_pair_t = real_iter_t.get_next()

#tf.reset_default_graph()
nn = net_3d_5.VAE_GAN1()



# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    saver = tf.train.Saver(max_to_keep=2000)


    time_start = timer()
    sess.run(real_iter.initializer)
    sess.run(real_iter_t.initializer)
    # training-loop

    loss_disc_real =  tf.Variable(0.0)
    loss_disc_fake =  tf.Variable(0.0)
    disc_acc =  tf.Variable(0.0)
    encode_loss =  tf.Variable(0.0)
    RC_loss =  tf.Variable(0.0)


    tf.summary.scalar("loss_disc_real", loss_disc_real)
    tf.summary.scalar("loss_disc_fake", loss_disc_fake)
    tf.summary.scalar("disc_acc", disc_acc)
    tf.summary.scalar("encode_loss", encode_loss)
    tf.summary.scalar("RC_loss", RC_loss)

    write_op = tf.summary.merge_all()

    writer_train = tf.summary.FileWriter(graph_folder+'/train/')
    writer_test = tf.summary.FileWriter(graph_folder+'/test/')

    n=0
    Disc_Acc = 1
    for batch in range(1000000):
            if n > real_data.N_batches-2:
                sess.run(real_iter.initializer)
                sess.run(real_iter_t.initializer)
                n=0
            n=n+1

            vx_real ,mp_real= sess.run(real_pair)
            fd ={nn.vx_real: vx_real, nn.mp_real: mp_real}

            if Disc_Acc<0.8:
                sess.run(nn.opti_D, feed_dict=fd)
            # optimization GAN
            sess.run(nn.opti_EC   , feed_dict=fd)

            Disc_Acc = sess.run(nn.Disc_Acc, feed_dict=fd)

            #update
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            sess.run(extra_update_ops)

            #Learrate
            new_learn_rate = sess.run(nn.new_learning_rate)
            if new_learn_rate > 0.00005:
                sess.run(nn.add_global)
            if batch %1 == 0:
                D_Real, D_Fake, L_tot,Disc_Acc,Enc_loss, reconstr_loss= sess.run([nn.D_loss_real,nn.D_loss_fake,nn.VAE_GAN_loss, nn.Disc_Acc, nn.encode_loss, nn.reconstr_loss],feed_dict=fd)
                print("D_Real {:04.2f}  D_Fake {:04.2f} L_tot {:04.2f},Disc_Acc {:04.2f} Enc_loss {:04.2f} reconstr_loss {:04.2f} ".format(D_Real, D_Fake,L_tot, Disc_Acc,Enc_loss, reconstr_loss))
            if batch %1 == 0:
                # #SAVE  TRAIN DATA
                fd = {loss_disc_real: D_Real,\
                loss_disc_fake: D_Fake,\
                disc_acc: Disc_Acc,\
                RC_loss: reconstr_loss,\
                encode_loss: Enc_loss}

                summary = sess.run(write_op,feed_dict = fd)
                writer_train.add_summary(summary, batch)
                writer_train.flush()
                # #SAVE  TEST DATA
                vx_real ,mp_real= sess.run(real_pair_t)
                fd ={nn.vx_real: vx_real, nn.mp_real: mp_real}
                D_Real, D_Fake, L_tot,Disc_Acc,Enc_loss, reconstr_loss= sess.run([nn.D_loss_real,nn.D_loss_fake,nn.VAE_GAN_loss, nn.Disc_Acc, nn.encode_loss, nn.reconstr_loss],feed_dict=fd)
                print("TEST")
                print("D_Real {:04.2f}  D_Fake {:04.2f} L_tot {:04.2f},Disc_Acc {:04.2f} Enc_loss {:04.2f} reconstr_loss {:04.2f} ".format(D_Real, D_Fake,L_tot, Disc_Acc,Enc_loss, reconstr_loss))
                #
                #SAVE  TEST DATA
                fd = {loss_disc_real: D_Real,\
                loss_disc_fake: D_Fake,\
                disc_acc: Disc_Acc,\
                RC_loss: reconstr_loss,\
                encode_loss: Enc_loss}
                summary = sess.run(write_op,feed_dict = fd)
                writer_test.add_summary(summary, batch)
                writer_test.flush()
                #

                saved_path = saver.save(sess, model_path + str(batch) + ".ckpt")
                saver.restore(sess,model_path + str(batch) + ".ckpt")
                print("Model Saved")
#                map_gen = sess.run(nn.mp_fake,feed_dict=fd)
#                res = {"inp":vx_real,"mp_real":mp_real,"mp_gen":map_gen}
#                with  open(test_res_folder + str(batch) + ".pkl", 'wb') as out_file:
#                    pickle.dump(res, out_file)
