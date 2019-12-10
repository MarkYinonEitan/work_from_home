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
from dataset_loader import EM_DATA_DISC_RANDOM
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

if os.path.isdir(out_fld):
    shutil.rmtree(out_fld, ignore_errors=True)

os.mkdir(out_fld)
os.mkdir(model_path)
os.mkdir(graph_folder)
os.mkdir(test_res_folder)





data_pairs = dataset_loader.read_list_file(data_fld+'list.txt')
all_pdbs = [x[0] for x in  data_pairs]



em_train = EM_DATA_DISC_RANDOM(data_fld,train_pdbs=all_pdbs[:20])
train_data = em_train.train_dataset.make_initializable_iterator()
trn = train_data.get_next()

em_test = EM_DATA_DISC_RANDOM(data_fld,train_pdbs=all_pdbs[21:23])
test_data = em_test.train_dataset.make_initializable_iterator()
tst = test_data.get_next()

nn = net_3d_1.DISC_V1()
disc_loss = nn.disc_loss


train_sum = [tf.summary.scalar('loss',disc_loss["loss"]),
tf.summary.scalar('acc',disc_loss["acc"])]
merged_train = tf.summary.merge(train_sum)

test_sum = [tf.summary.scalar('loss',disc_loss["loss"]),
tf.summary.scalar('acc',disc_loss["acc"])]
merged_test = tf.summary.merge(test_sum)

# open session and initialize all variables
config = tf.ConfigProto(log_device_placement=True)
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
        for batch in range(em_train.N_batches-1):
            feature,x_label = sess.run(trn)

            loss_d, acc_d,log_train, _= sess.run([disc_loss["loss"], disc_loss["acc"], merged_train,nn.D_optim],\
                        feed_dict={nn.x: feature, nn.x_label: x_label, nn.keep_prob: 0.8, nn.isTrain: True})
            sess.run(nn.clip)

            if batch % 10 == 0:
                time_end = timer()
                print("EPOCH {} of {} ## BATCH {} of {} TIME {} ### BATCH_SIZE:{} IS_GPU: {} ".\
                format(epoch_num,N_EPOCHS,batch,em_train.N_batches,time_end - time_start,BATCH_SIZE,' 4 GPUS'))
                time_start = timer()
                writer.add_summary(log_train,batch)
            if batch % 100 == 0:
                feature,x_label = sess.run(tst)
                loss_d_test, acc_d_test,log_test = sess.run([disc_loss["loss"], disc_loss["acc"], merged_test],\
                feed_dict={nn.x: feature, nn.x_label: x_label, nn.keep_prob: 0.8, nn.isTrain: False})
                writer.add_summary(log_test,batch)

                print("D Loss : train {} : test {}:".format(loss_d,loss_d_test))
                print("Disc Acc : train {} : test {}:".format(acc_d,acc_d_test))

                saved_path = saver.save(sess, model_path + str(batch) + ".ckpt")
                print("Model Saved")
