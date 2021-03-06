from tensorflow.python.client import device_lib
import tensorflow as tf
import matplotlib.pyplot as plt
import dataset_loader
import numpy as np
from net_3d_5 import get_net_by_string
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS


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


def run_disc_on_test_points(test_points, disc_net_str, disc_wights_file):

    tf.reset_default_graph()
    disc_res = np.zeros(len(test_points))
    disc_net  = get_net_by_string(disc_net_str)

    # open session and initialize all variables
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        saver.restore(sess, disc_wights_file)



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



            if n == BATCH_SIZE and True:
                disc_out_gen = sess.run(disc_net.D_real,feed_dict={disc_net.vx_real: inp_vx, disc_net.mp_real: out_mp})
                print(np.mean(disc_out_gen))
                for i in range(BATCH_SIZE-1):
                    disc_res[kn[i]] =disc_out_gen[i]
                    test_points[kn[i]].out = test_points[kn[i]].out*0+disc_res[kn[i]]
                print("DEBUG 21", disc_res.shape, np.sum(disc_res>0.5),np.sum(disc_res<0.5))


    return test_points


#def display_point(p):
