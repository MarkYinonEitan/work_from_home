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
importlib.reload(dataset_loader)
from dataset_loader import EM_DATA
from dataset_loader import BATCH_SIZE, NBOX_IN,NBOX_OUT,N_CHANNELS
import net_3d_5
import utils_project
import train_vaegan



def old_func():

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

    test_points_inp = utils_project.getTestPoints(em_data)
    test_points_disc = utils_project.disc_test_points(test_points_inp, disc_net, disc_weights_file)


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


    return



def run_discriminator(inp_data_fld, pdb_id, out_file_name,disc_net_str,disc_weights_file):
    em_data = EM_DATA(inp_data_fld,train_pdbs=[pdb_id], is_random = False)
    test_points = utils_project.getTestPoints(em_data)
    res_disc = utils_project.run_disc_on_test_points(test_points, disc_net_str, disc_weights_file)
    out_matrx = utils_project.tstpoint2mtrx(test_points)
    np.save(out_file_name,out_matrx)

def assert_vx_size_and_resolution(vx_size,res):
    if (vx_size != dataset_loader.VOX_SIZE ) or (res != dataset_loader.RESOLUTION ):
        raise Exception("VX_SIZE or RES uncorrect")


if __name__ == "__main__":
    list_file   = "/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/list_3A.txt"
    vx_folder   = "/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/vx_data/"
    out_folder  = "/specific/netapp5_2/iscb/wolfson/Mark/data/AAcryoGaN3/output/"
    net_str     = "gan_v1"
    n_epochs    = 30
    resolution  = 3.0
    vx_size     = 1.0

    print("list_file:",list_file)
    print("vx_folder:",vx_folder)
    print("out_folder:",out_folder)
    print("net_str:",net_str)
    print("n_epochs:",n_epochs)
    print("resolution:",resolution)
    print("vx_size:",vx_size)

    assert_vx_size_and_resolution(vx_size,resolution)
    real_data_train, real_data_test = train_vaegan.get_data_for_train(vx_fold = vx_folder,list_file = list_file)
    print("DEBUG 1102",real_data_train.N_batches)
    train_vaegan.run_training(real_data_train,real_data_test, net_string = net_str,out_fld = out_folder)
