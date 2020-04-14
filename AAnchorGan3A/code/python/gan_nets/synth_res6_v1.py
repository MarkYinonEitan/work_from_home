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
out_fld = base_data_folder + "/results/synth_res6/"

model_path = out_fld+'/network_test/'
graph_folder = out_fld+'/graphs/'
test_res_folder = out_fld + '/tests/'

if os.path.isdir(out_fld):
    shutil.rmtree(out_fld, ignore_errors=True)

os.mkdir(out_fld)
os.mkdir(model_path)
os.mkdir(graph_folder)
os.mkdir(test_res_folder)





data_pairs = dataset_loader.read_list_file(data_fld+'list.txt')
all_pdbs = [x[0] for x in  data_pairs]

print("DEBUG 6666")
print(DEVICE_GPU)

#with tf.device("/gpu:2"):
#with 5 as aaa:

em_train = EM_DATA(data_fld,train_pdbs=all_pdbs[:20])
em_test = EM_DATA(data_fld,train_pdbs=all_pdbs[21:24])

train_data = em_train.train_dataset.make_initializable_iterator()
test_data = em_test.train_dataset.make_initializable_iterator()

trn = train_data.get_next()
tst = test_data.get_next()

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_IN, NBOX_IN, NBOX_IN,N_CHANNELS ])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_OUT, NBOX_OUT, NBOX_OUT,1 ])
keep_prob = tf.placeholder(dtype=tf.float32)
isTrain = tf.placeholder(dtype=tf.bool)

nn = net_3d_1.get_net_graph(x,y,keep_prob,isTrain)


##
train_sum = [tf.summary.scalar('D_loss_Train',nn["D_loss"]),
tf.summary.scalar('G_loss_Train',nn["G_loss"]),
tf.summary.scalar('VAE_L_Train',nn["VAE_L"]),
tf.summary.scalar('KL_DIV_Train',nn["mean_KL"]),
tf.summary.scalar('Recostr_loss_Train',nn["mean_recon"])]
merged_train = tf.summary.merge(train_sum)

test_sum = [tf.summary.scalar('D_loss_TEST',nn["D_loss"]),
tf.summary.scalar('G_loss_TEST',nn["G_loss"]),
tf.summary.scalar('VAE_L_TEST',nn["VAE_L"]),
tf.summary.scalar('KL_DIV_TEST',nn["mean_KL"]),
tf.summary.scalar('Recostr_loss_TEST',nn["mean_recon"])]
merged_test = tf.summary.merge(test_sum)


# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True






with tf.Session(config=config) as sess:
    #sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter(graph_folder, sess.graph)
    saver = tf.train.Saver()

    time_start = timer()
    utils.run_to_check_if_usign_gpu(sess)
    for epoch_num in range(N_EPOCHS):
        sess.run(train_data.initializer)
        sess.run(test_data.initializer)
        # training-loop
        for batch in range(em_train.N_batches):
            image_in,image_out = sess.run(trn)
            for _ in range(4):
                sess.run(nn["D_optim"], feed_dict={x: image_in, y: image_out, keep_prob: 0.8, isTrain: True})
                sess.run(nn["clip"])
            loss_d_, loss_g_, _VAE_loss, _KL_divergence, _reconstruction_loss, sum_train,\
            _, _, _ = sess.run([nn["D_loss"], nn["G_loss"], nn["VAE_L"], nn["mean_KL"],\
            nn["mean_recon"], merged_train, nn["D_optim"], nn["G_optim"], nn["E_optim"]],\
                        feed_dict={x: image_in, y: image_out, keep_prob: 0.8, isTrain: True})
            sess.run(nn["clip"])

            if batch % 10 == 0:
                time_end = timer()
                print("EPOCH {} of {} ## BATCH {} of {} TIME {} ### BATCH_SIZE:{} IS_GPU: {} ".\
                format(epoch_num,N_EPOCHS,batch,em_train.N_batches,time_end - time_start,BATCH_SIZE,' 4 GPUS'))
                time_start = timer()
                writer.add_summary(sum_train,batch)
            if batch % 100 == 99:
                print("DEBUG 33344 100 {}".format(batch))

                image_in,image_out = sess.run(tst)
                print("DEBUG 33344 101 {}".format(batch))

                loss_d_test, loss_g_test, _VAE_loss_test, _KL_divergence_test, _reconstruction_loss_test,\
                sum_test = sess.run([nn["D_loss"], nn["G_loss"], nn["VAE_L"], nn["mean_KL"],\
                nn["mean_recon"], merged_test],feed_dict={x: image_in, y: image_out, keep_prob: 0.8, isTrain: False})
                print("DEBUG 33344 102 {}".format(batch))

                writer.add_summary(sum_test,batch)

                print("D Loss : train {} : test {}:".format(loss_d_,loss_d_test))
                print("G Loss: train {} : test {}:".format(loss_g_,loss_g_test))
                print("VAE loss: train {} : test {}:".format(_VAE_loss,_VAE_loss_test))
                print("KL divergence : train {} : test {}:".format(_KL_divergence,_KL_divergence_test))
                print("reconstruction_loss: train {} : test {}:".format( _reconstruction_loss,_reconstruction_loss_test))
                print("###########")
                G = sess.run(nn["G_z"]["o"], feed_dict={x: image_in, y: image_out, keep_prob: 0.8, isTrain: False})
                np.save(test_res_folder + str(batch) + ".npy", G)
                if batch % 1000 == 0:
                    saver.save(sess, model_path + str(batch) + ".ckpt")
