import os
import sys
import importlib
import numpy as np
import shutil
import tensorflow as tf
from timeit import default_timer as timer

python_path = '//specific//netapp5_2//iscb//wolfson/Mark//git//work_from_home/AAcryoGAN3/code/python'
sys.path.append(python_path)

import dataset_loader
from dataset_loader import EM_DATA
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
import utils, utils_project

def getTestPoints(em_data):
    test_points = [None]*em_data.N_train
    for n, p in enumerate(em_data.train_points):
        test_points[n]= dataset_loader.TestBox(p)
    return test_points

def tstpoint2mtrx(tstp):
    out_mtrx = np.zeros(tstp[0].p.out.shape)
    n_mtrx = np.ones(tstp[0].p.out.shape)
    NN = NBOX_OUT
    for tp in tstp:
        res_map = tp.out;#dataset_loader.adjust_mean_std(tp.out, mean = tp.p.mean, sigma = tp.p.sigma)

        I,J,K = tp.p.ijk

        out_mtrx[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1] = \
        out_mtrx[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1] + res_map[:,:,:,0]

        n_mtrx[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1] = n_mtrx[I-NN//2:I+NN//2+1,J-NN//2:J+NN//2+1,K-NN//2:K+NN//2+1] + 1

    return out_mtrx/n_mtrx

def disc_test_points(test_points, disc_net_file):
    tf.reset_default_graph()
    disc_net = net_3d_5.DISC_V1()
    disc_res = np.zeros(len(test_points))

    # open session and initialize all variables
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        saver.restore(sess, disc_net_file)



        kn={};
        for k, tp  in enumerate(test_points):

            if k%BATCH_SIZE == 0:
                inp_vx = np.zeros([BATCH_SIZE,NBOX_IN,NBOX_IN,NBOX_IN,N_CHANNELS])
                out_mp = np.zeros([BATCH_SIZE,NBOX_OUT,NBOX_OUT,NBOX_OUT,1])
                n = 0

            inp_vx[n,:,:,:,:] = dataset_loader.get_inp_map(tp.p)
            out_mp[n,:,:,:,:] = tp.out

            kn[n]=k
            n=n+1

    #        map_out = dataset_loader.get_out_map(p)
    #        test_points[k].out = map_out


            if n == BATCH_SIZE and True:
                disc_out_gen = sess.run(disc_net.D_real,feed_dict={disc_net.vx_real: inp_vx, disc_net.mp_real: out_mp})
                print(np.mean(disc_out_gen))
                for i in range(BATCH_SIZE-1):
                    disc_res[kn[i]] =disc_out_gen[i]
                print("DEBUG 21", disc_res.shape, np.sum(disc_res>0.5),np.sum(disc_res<0.5))

        return disc_res

    out_matrx = tstpoint2mtrx(test_points)

    out_file_name = out_fld +'test_'+pdb_id +'.npy'

    np.save(out_file_name,out_matrx)
    return


def create_one_map(pdb_file, inp_data_fld,net_string, vae_gan_file,out_file_name):

    em_data = EM_DATA(inp_data_fld,train_pdbs=[pdb_file], is_random = False)
    test_points = getTestPoints(em_data)

    print("DEBUG 3238204", pdb_file)
    print(len(test_points))

    vae_gan = utils_project.get_net_by_string(net_string)

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
                    test_points[kn[i]].out =map_out[i,:,:,:,:]

    out_matrx = tstpoint2mtrx(test_points)
    np.save(out_file_name,out_matrx)

if __name__ == "__main__":
    list_file   = sys.argv[1]
    vx_folder   = sys.argv[2]
    out_folder  = sys.argv[3]
    net_str     = sys.argv[4]
    vae_gan_file = sys.argv[5]
    resolution  = np.float(sys.argv[6])
    vx_size     = np.float(sys.argv[7])

    print("list_file:",list_file)
    print("vx_folder:",vx_folder)
    print("out_folder:",out_folder)
    print("vae_gan_file:",vae_gan_file)
    print("net_str:",net_str)
    print("resolution:",resolution)
    print("vx_size:",vx_size)

    utils_project.assert_vx_size_and_resolution(vx_size,resolution)

    pairs = utils_project.read_list_file(list_file)
    for pair in pairs:
        pdb_file = pair["pdb_file"]
        f_out = dataset_loader.get_file_names(pdb_file,out_folder)
        out_file_name = f_out["GAN_NPY"]
        tf.keras.backend.clear_session()
        create_one_map(pdb_file, vx_folder,net_str, vae_gan_file,out_file_name)
