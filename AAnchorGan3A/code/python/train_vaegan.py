

import os
import sys
import importlib
import numpy as np
import shutil
import pickle
import tensorflow as tf
import time
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path + '/../python/'
sys.path.append(python_path)
import copy



import dbloader
importlib.reload(dbloader)
from dbloader import  EM_DATA
from dbloader import VX_BOX_SIZE, MAP_BOX_SIZE, N_CHANNELS, BATCH_SIZE
import utils, utils_project
from utils import get_available_gpus

def set_out_folders(out_fld = None):
        ## organize folders
        model_path = out_fld+'/network_test/'
        graph_folder = out_fld+'/graphs/'
        test_res_folder = out_fld + '/tests/'
        if os.path.isdir(out_fld):
            shutil.rmtree(out_fld, ignore_errors=True)

        os.mkdir(out_fld)
        os.mkdir(model_path)
        os.mkdir(graph_folder)
        os.mkdir(test_res_folder)

        return model_path, graph_folder, test_res_folder

def run_training(real_data_train,real_data_test, net_string = 'None',out_fld = None):

    model_path, graph_folder, test_res_folder = set_out_folders(out_fld = out_fld)

    #tf.reset_default_graph()
    nn = utils_project.get_net_by_string(net_string)

    real_iter_train = real_data_train.train_dataset.make_initializable_iterator()
    real_pair_train = real_iter_train.get_next()

    real_iter_test = real_data_test.train_dataset.make_initializable_iterator()
    real_pair_test = real_iter_test.get_next()


    # open session and initialize all variables
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()


        saver = tf.train.Saver(max_to_keep=2000)


        sess.run(real_iter_test.initializer)
        sess.run(real_iter_train.initializer)
        # training-loop

        loss_disc_real =  tf.Variable(0.0)
        loss_disc_fake =  tf.Variable(0.0)
        disc_acc =  tf.Variable(0.0)
        encode_loss =  tf.Variable(0.0)
        RC_loss =  tf.Variable(0.0)
        Mean_Map =  tf.Variable(0.0)
        Sigma_Map =  tf.Variable(0.0)




        tf.summary.scalar("loss_disc_real", loss_disc_real)
        tf.summary.scalar("loss_disc_fake", loss_disc_fake)
        tf.summary.scalar("disc_acc", disc_acc)
        tf.summary.scalar("encode_loss", encode_loss)
        tf.summary.scalar("RC_loss", RC_loss)
        tf.summary.scalar("Mean_Map", Mean_Map)
        tf.summary.scalar("Sigma_Map", Sigma_Map)

        write_op = tf.summary.merge_all()

        writer_train = tf.summary.FileWriter(graph_folder+'/train/')
        writer_test = tf.summary.FileWriter(graph_folder+'/test/')

        n=0
        Disc_Acc = 1
        for batch in range(1000000):
                if n > real_data_train.N_batches-2:
                    sess.run(real_iter_train.initializer)
                    sess.run(real_iter_test.initializer)
                    n=0
                n=n+1

                vx_real ,mp_real= sess.run(real_pair_train)
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


                #print train results every 10 batches
                if batch %10 == 0:
                    D_Real, D_Fake, L_tot,Disc_Acc,Enc_loss, reconstr_loss, mean_map, sigma_map= sess.run([nn.D_loss_real,nn.D_loss_fake,nn.VAE_GAN_loss, nn.Disc_Acc, nn.encode_loss, nn.reconstr_loss, nn.mp_fake_mean, nn.mp_fake_sigma],feed_dict=fd)
                    print("D_Real {:04.2f}  D_Fake {:04.2f} L_tot {:04.2f},Disc_Acc {:04.2f} Enc_loss {:04.2f} reconstr_loss {:04.2f} Mean {:04.2f} Sigma {:04.2f} ".format(D_Real, D_Fake,L_tot, Disc_Acc,Enc_loss, reconstr_loss, mean_map, sigma_map))
                    print("BATCH: ",batch, "  TIME:" , time.ctime())

                #print test results every 100 batches
                if batch %100 == 0:
                    # #SAVE  TRAIN DATA
                    fd = {loss_disc_real: D_Real,\
                    loss_disc_fake: D_Fake,\
                    disc_acc: Disc_Acc,\
                    RC_loss: reconstr_loss,\
                    encode_loss: Enc_loss,\
                    Mean_Map : mean_map,\
                    Sigma_Map :sigma_map}


                    summary = sess.run(write_op,feed_dict = fd)
                    writer_train.add_summary(summary, batch)
                    writer_train.flush()
                    # #SAVE  TEST DATA
                    vx_real ,mp_real= sess.run(real_pair_test)
                    fd ={nn.vx_real: vx_real, nn.mp_real: mp_real}
                    D_Real, D_Fake, L_tot,Disc_Acc,Enc_loss, reconstr_loss, mean_map, sigma_map= sess.run([nn.D_loss_real,nn.D_loss_fake,nn.VAE_GAN_loss, nn.Disc_Acc, nn.encode_loss, nn.reconstr_loss, nn.mp_fake_mean, nn.mp_fake_sigma],feed_dict=fd)
                    print("TEST")
                    print("D_Real {:04.2f}  D_Fake {:04.2f} L_tot {:04.2f},Disc_Acc {:04.2f} Enc_loss {:04.2f} reconstr_loss {:04.2f} Mean {:04.2f} Mean {:04.2f} ".format(D_Real, D_Fake,L_tot, Disc_Acc,Enc_loss, reconstr_loss, mean_map, sigma_map))

                    #
                    #SAVE  TEST DATA
                    fd = {loss_disc_real: D_Real,\
                    loss_disc_fake: D_Fake,\
                    disc_acc: Disc_Acc,\
                    RC_loss: reconstr_loss,\
                    encode_loss: Enc_loss,\
                    Mean_Map : mean_map,\
                    Sigma_Map :sigma_map}

                    summary = sess.run(write_op,feed_dict = fd)
                    writer_test.add_summary(summary, batch)
                    writer_test.flush()
                    #

                    saved_path = saver.save(sess, model_path + str(batch) + ".ckpt")
                    saver.restore(sess,model_path + str(batch) + ".ckpt")
                    print("Model Saved")


def run_test(test_data, net_string = 'None', vae_gan_file = 'None'):


    #Initialize Netowrk
    vae_gan = utils_project.get_net_by_string(net_string)

    #initalize ouputs
    res_dict = copy.deepcopy(test_data.train_data_dict["data"])
    gan_maps = [np.zeros([MAP_BOX_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE])]*test_data.N_train

    #Load one file
    real_iter_test = test_data.train_dataset.make_initializable_iterator()
    real_pair_test = real_iter_test.get_next()

    # open session and initialize all variables
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        saver = tf.train.Saver()

        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver.restore(sess, vae_gan_file)

        sess.run(real_iter_test.initializer)

        gan_maps = [np.zeros([MAP_BOX_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE])]*test_data.N_train
        for k in range(test_data.N_train):

            if k%BATCH_SIZE == 0:
                print(k, test_data.N_train)
                inp_batch = np.zeros([BATCH_SIZE,VX_BOX_SIZE,VX_BOX_SIZE,VX_BOX_SIZE,N_CHANNELS])
                out_batch = np.zeros([BATCH_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE,MAP_BOX_SIZE,1])
                n = 0
                kn = [-1]*BATCH_SIZE

            (vx_map, em_map) = dbloader.get_data_point(test_data.train_data_dict,k)
            inp_batch[n,:,:,:,:] = vx_map
            out_batch[n,:,:,:,:] = em_map

            kn[n]=k
            n=n+1

            if n==BATCH_SIZE or k==test_data.N_train:
                map_out, r_loss, gan_disc_res = sess.run([vae_gan.mp_fake, vae_gan.reconstr_loss, vae_gan.D_loss_fake],feed_dict={vae_gan.vx_real: inp_batch, vae_gan.mp_real: out_batch})
                print(r_loss, np.mean(out_batch),np.mean(map_out),np.std(out_batch),np.std(map_out))
                for i in range(n-1):
                    gan_maps[kn[i]]=np.squeeze(map_out[i,:,:,:,:])
                    res_dict[kn[i]]["RC_loss"] = r_loss
                    res_dict[kn[i]]["gan_disc_res"] = gan_disc_res
                    res_dict[kn[i]]["MAP_SOURCE"] = "GAN"

    return gan_maps, res_dict
