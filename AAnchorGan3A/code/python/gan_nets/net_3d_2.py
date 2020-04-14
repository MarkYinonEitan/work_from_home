import os, time, sys, itertools
import numpy as np
import matplotlib
import tensorflow as tf

from ops import batch_normal, de_conv, conv3d, fully_connect, lrelu


#get current directory
if '__file__' in locals():
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    dir_path = os.getcwd()
python_path = dir_path
sys.path.append(python_path)
import dataset_loader
from dataset_loader import read_list_file, get_file_names,VX_FILE_SUFF
from dataset_loader import VOX_SIZE, RESOLUTION, N_SAMPLS_FOR_1V3
from dataset_loader import N_CHANNELS, BATCH_SIZE,NBOX_OUT, NBOX_IN

matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 10
D_lr = 5e-5
G_lr = 1e-4
train_epoch = 20
n_latent = 100
alpha_1 = 5
alpha_2 = 5e-4

Input_Cube_Size = 8
Output_Cube_Size = 5
Nchannels = 4

root="./test1/"


class ENC_V1():
    def __init__(self):

        self.learn_rate_init = 0.03


        #Learning Rate
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=self.global_step, decay_steps=10000,
                                                   decay_rate=0.98)


        self.log_vars = []

        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_OUT, NBOX_OUT, NBOX_OUT ,1])
        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_IN, NBOX_IN, NBOX_IN ,N_CHANNELS])
        self.vx_real = vx_real


        #encode
        self.z_mean, self.z_sigm = self.Encode(self.vx_real)
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)

        #For encode
        self.encode_loss = self.kl_loss/(128*BATCH_SIZE)

        t_vars = tf.trainable_variables()

        self.log_vars.append(("e_loss", self.encode_loss))

        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        #trainers
        self.trainer_E = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        self.gradients_E = self.trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
        self.opti_E = self.trainer_E.apply_gradients(self.gradients_E)


    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def Encode(self,x):

        with tf.variable_scope('encode') as scope:

            conv1 = tf.nn.relu(batch_normal(conv3d(x, output_dim=32, k_dwh=[1, 1, 1],  d_dwh=(1, 1, 1),  pad='SAME', name='e_c1'), scope='e_bn1'))
            conv2 = tf.nn.relu(batch_normal(conv3d(conv1, output_dim=128, k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID',name='e_c2'), scope='e_bn2'))
            conv3 = tf.nn.relu(batch_normal(conv3d(conv2 , output_dim=256,k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1), pad='VALID', name='e_c3'), scope='e_bn3'))

            conv3 = tf.reshape(conv3, [BATCH_SIZE, -1])
            fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=256, scope='e_f1'), scope='e_bn4'))
            z_mean = fully_connect(fc1 , output_size=128, scope='e_f2')
            z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

            return z_mean, z_sigma



class DISC_V1():
    def __init__(self):

        self.learn_rate_init = 0.03


        #Learning Rate
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=self.global_step, decay_steps=10000,
                                                   decay_rate=0.98)


        d_scale_factor = 0.25

        self.log_vars = []



        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_OUT, NBOX_OUT, NBOX_OUT ,1])
        mp_fake = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_OUT, NBOX_OUT, NBOX_OUT ,1])

        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_IN, NBOX_IN, NBOX_IN ,N_CHANNELS])
        vx_fake = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_IN, NBOX_IN, NBOX_IN , N_CHANNELS])

        # inputs
        self.mp_real = mp_real
        self.mp_fake = mp_fake

        self.vx_real = vx_real
        self.vx_fake = vx_fake

        self.mp_random = tf.random_normal(shape=self.mp_real.get_shape())
        self.vx_random = tf.random_normal(shape=self.vx_real.get_shape())

        #networks
        self.l_x,  self.D_pro_logits = self.discriminate(self.mp_real,self.vx_real, False)
        self.l_x_tilde, self.De_pro_tilde = self.discriminate(self.mp_random,self.vx_random,True)
        _, self.G_pro_logits = self.discriminate(self.mp_fake,self.vx_fake, True)



        self.D_real_loss = tf.reduce_mean(\
                            tf.nn.sigmoid_cross_entropy_with_logits(\
                            labels=tf.ones_like(self.D_pro_logits)- d_scale_factor ,logits=self.D_pro_logits))
        self.D_fake_loss = tf.reduce_mean(\
                            tf.nn.sigmoid_cross_entropy_with_logits(\
                            labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.De_pro_tilde), logits=self.De_pro_tilde))

        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_tilde_loss

        t_vars = tf.trainable_variables()


        self.log_vars.append(("discriminator_loss", self.D_loss))

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        #trainers
        self.trainer_D = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        self.gradients_D = self.trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        self.opti_D = self.trainer_D.apply_gradients(self.gradients_D)



    def discriminate(self, mp, vx, reuse=False):
        paddings =  tf.constant([[0,0],
                                 [(NBOX_IN-NBOX_OUT)//2,(NBOX_IN-NBOX_OUT)//2],\
                                 [(NBOX_IN-NBOX_OUT)//2,(NBOX_IN-NBOX_OUT)//2],\
                                 [(NBOX_IN-NBOX_OUT)//2,(NBOX_IN-NBOX_OUT)//2],\
                                 [0,0]])
        mp_pad = tf.pad(mp,paddings,mode='CONSTANT')
        x_var= tf.concat([vx, mp_pad], 4)

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.relu(conv3d(x_var, output_dim=32, k_dwh=[1, 1, 1],  d_dwh=(1, 1, 1), pad='SAME', name='dis_conv1'))
            conv2= tf.nn.relu(batch_normal(conv3d(conv1, output_dim=32,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= tf.nn.relu(batch_normal(conv3d(conv2, output_dim=32,k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1), pad='VALID', name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            conv4 = conv3d(conv3, output_dim=128,k_dwh=[2, 2, 2],  d_dwh=(2, 2, 2),pad='VALID', name='dis_conv4')
            middle_conv = conv4
            conv4= tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
            conv4= tf.reshape(conv4, [BATCH_SIZE, -1])

            fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=32, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
            output = fully_connect(fl , output_size=1, scope='dis_fully2')

            return middle_conv, output

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


def encoder(x, keep_prob=0.5, isTrain=True):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):  #B*9*9*9*5
        enc={}

        conv1 = tf.layers.conv3d(x, 128, [2, 2, 2], strides=(1, 1, 1), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())  # 32 * 32 * 128
        lrelu1 = tf.nn.elu(conv1)
        enc["c1"] = conv1
        enc["l1"] = lrelu1
        enc["out1_shape"] = [BATCH_SIZE,9,9,9,128]

        conv2 = tf.layers.conv3d(lrelu1, 256, [4, 4, 4], strides=(2, 2, 2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())  # 16 * 16 *256
        lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))
        enc["c2"] = conv2
        enc["l2"] = lrelu2
        enc["out2_shape"] = [BATCH_SIZE,5,5,5,256]

        conv3 = tf.layers.conv3d(lrelu2, 512, [4, 4, 4], strides=(2, 2, 2), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())  # 8 * 8 * 512
        lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))
        enc["c3"] = conv3
        enc["l3"] = lrelu3
        enc["out3_shape"] = [BATCH_SIZE,3,3,3,512]

        conv4 = tf.layers.conv3d(lrelu3, 100, [3, 3, 3], strides=(3, 3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())  # 4 * 4 * 1024
        lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))
        enc["c4"] = conv4
        enc["l4"] = lrelu4
        enc["out4shape"] = [BATCH_SIZE,1,1,1,100]

        #conv5 = tf.layers.conv3d(lrelu3, 32, [4, 4, 4], strides=(1, 1, 1), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer())  # 1 * 1 * 32
        #lrelu5 = tf.nn.elu(tf.layers.batch_normalization(conv5, training=isTrain))

        x = tf.nn.dropout(lrelu3, keep_prob)
        x = tf.contrib.layers.flatten(x)
        z_mu = tf.layers.dense(x, units=n_latent)
        z_sig = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = z_mu + tf.multiply(epsilon, tf.exp(z_sig))

        enc["z"]=z
        enc["z_shape"]=[BATCH_SIZE,n_latent]
        enc["z_mu"]=z_mu
        enc["z_mu_shape"]=[]
        enc["z_sig"]=z_sig
        enc["z_sigshape"]=[]

        return enc


def generator(x, isTrain=True):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

        gnr = {}

        # 1st hidden layer
        conv1 = tf.layers.conv3d_transpose(x, 512, [2, 2, 2], strides=(2, 2, 2), padding='valid', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 2, 2, 2, 256)
        lrelu1 = tf.nn.elu(tf.layers.batch_normalization(conv1, training=isTrain))
        gnr["c1"] = conv1
        gnr["l1"] = lrelu1
        gnr["out1_shape"] = [BATCH_SIZE,3,3,3,512]

        #2nd hidden layer
        conv2 = tf.layers.conv3d_transpose(lrelu1, 256, [2, 2, 2], strides=(1, 1, 1), padding='valid', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer()) # (-1, 4, 4, 4, 128)
        lrelu2 = tf.nn.elu(tf.layers.batch_normalization(conv2, training=isTrain))
        gnr["c2"] = conv2
        gnr["l2"] = lrelu2
        gnr["out2shape"] = [BATCH_SIZE,3,3,3,256]

        # 3rd hidden layer
        conv3 = tf.layers.conv3d_transpose(lrelu2, 128, [3, 3, 3], strides=(1, 1, 1), padding='valid', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 8, 8, 8, 64)
        lrelu3 = tf.nn.elu(tf.layers.batch_normalization(conv3, training=isTrain))
        gnr["c3"] = conv3
        gnr["l3"] = lrelu3
        gnr["out3shape"] = [BATCH_SIZE,5,5,5,128]

        # 4rd hidden layer
        conv4 = tf.layers.conv3d_transpose(lrelu3, 1, [1, 1, 1], strides=(1, 1, 1), padding='valid', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 8, 8, 8, 64)
        lrelu4 = tf.nn.elu(tf.layers.batch_normalization(conv4, training=isTrain))
        gnr["c4"] = conv4
        gnr["out4shape"] = [BATCH_SIZE,5,5,5,1]

        # output layer
        #conv5 = tf.layers.conv3d_transpose(lrelu4, 1, [4, 4, 4], strides=(2, 2, 2), padding='same', use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())  # (-1, 32, 32, 32, 1)

        # output layer
        o = tf.nn.tanh(conv4)
        gnr["o"]=o

        return gnr


def get_net_graph(xx,yy,keep_prob,isTrain):

    x_image = xx#tf.placeholder(tf.float32, shape=(BATCH_SIZE, NBOX, NBOX,NBOX, N_CHANNELS))
    x_3D = yy#tf.placeholder(tf.float32, shape=(BATCH_SIZE, NBOX, NBOX, NBOX, 1))

    #keep_prob = tf.placeholder(dtype=tf.float32)
    #isTrain = tf.placeholder(dtype=tf.bool)


    net_graph={}
    net_graph["x_image"] = x_image
    net_graph["x_3D"] = x_3D
    net_graph["keep_prob"] = keep_prob
    net_graph["isTrain"] = isTrain



    # networks : encoder
    enc_layers = encoder(x_image, keep_prob, isTrain)
    net_graph["enc"] = enc_layers
    z, z_mu, z_sig = (enc_layers["z"], enc_layers["z_mu"], enc_layers["z_sig"])
    z = tf.reshape(z, (-1, 1, 1, 1, n_latent))

    net_graph["z"] = z
    net_graph["z_mu"] = z_mu
    net_graph["z_sig"] = z_sig

    # networks : generator
    net_graph["G_z"] = generator(z,isTrain)
    G_z = net_graph["G_z"]["o"]



    # networks : discriminator
    net_graph['dsc_real']= discriminator(x_3D, isTrain)
    net_graph['dsc_fake']= discriminator(G_z, isTrain)

    D_real,D_real_logits = (net_graph['dsc_real']["o"],net_graph['dsc_real']["logits"])
    D_fake,D_fake_logits = (net_graph['dsc_fake']["o"],net_graph['dsc_fake']["logits"])

    #D_fake, D_fake_logits = discriminator(G_z, isTrain)

    reconstruction_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(G_z, (BATCH_SIZE, NBOX_OUT**3)), tf.reshape(x_3D, (BATCH_SIZE, NBOX_OUT**3))),1)/NBOX_OUT**3
    net_graph["R_L"] = reconstruction_loss

    KL_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu ** 2 - tf.exp(2.0 * z_sig), 1)
    net_graph["KL_D"] = KL_divergence
    mean_KL = tf.reduce_sum(KL_divergence)
    mean_recon = tf.reduce_sum(reconstruction_loss)

    net_graph["mean_KL"] = mean_KL
    net_graph["mean_recon"] = mean_recon

    VAE_loss = tf.reduce_mean(alpha_1 * KL_divergence + alpha_2 * reconstruction_loss)

    net_graph["VAE_L"] = VAE_loss

    D_loss_real = tf.reduce_mean(D_real_logits)
    D_loss_fake = tf.reduce_mean(D_fake_logits)
    D_loss = D_loss_real - D_loss_fake
    G_loss = -tf.reduce_mean(D_fake_logits)

    net_graph["D_loss" ]=D_loss
    net_graph["G_loss" ]=G_loss

    # trainable variables for each network
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    E_vars = [var for var in T_vars if var.name.startswith('encoder')]

    net_graph["T_vars"] = T_vars
    net_graph["D_vars"] = D_vars
    net_graph["G_vars"] = G_vars
    net_graph["E_vars"] = E_vars

    clip = [p.assign(tf.clip_by_value(p, -0.5, 0.5)) for p in D_vars]
    net_graph["clip"] = clip

    # optimizer for each network
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_optim = tf.train.RMSPropOptimizer(D_lr).minimize(-D_loss, var_list=D_vars)
        G_optim = tf.train.RMSPropOptimizer(G_lr).minimize(G_loss, var_list=G_vars)
        E_optim = tf.train.AdamOptimizer(G_lr).minimize(VAE_loss, var_list=E_vars)

    net_graph["D_optim"] = D_optim
    net_graph["G_optim"] = G_optim
    net_graph["E_optim"] = E_optim


    return net_graph


if __name__ == "__main__":



    # networks : discriminator
    D_real, D_real_logits = discriminator(x_3D, isTrain)
    D_fake, D_fake_logits = discriminator(G_z, isTrain)

    # loss for each network

    reconstruction_loss = tf.reduce_sum(tf.squared_difference(tf.reshape(G_z, (-1, 32 * 32 * 32)), tf.reshape(x_3D, (-1, 32 * 32 * 32))),1)
    KL_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_sig - z_mu ** 2 - tf.exp(2.0 * z_sig), 1)
    mean_KL = tf.reduce_sum(KL_divergence)
    mean_recon = tf.reduce_sum(reconstruction_loss)

    VAE_loss = tf.reduce_mean(alpha_1 * KL_divergence + alpha_2 * reconstruction_loss)


    D_loss_real = tf.reduce_mean(D_real_logits)
    D_loss_fake = tf.reduce_mean(D_fake_logits)
    D_loss = D_loss_real - D_loss_fake
    G_loss = -tf.reduce_mean(D_fake_logits)
    # sub_loss = G_loss + VAE_loss


    # trainable variables for each network
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    E_vars = [var for var in T_vars if var.name.startswith('encoder')]

    clip = [p.assign(tf.clip_by_value(p, -0.5, 0.5)) for p in D_vars]


    # optimizer for each network
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_optim = tf.train.RMSPropOptimizer(D_lr).minimize(-D_loss, var_list=D_vars)
        G_optim = tf.train.RMSPropOptimizer(G_lr).minimize(G_loss, var_list=G_vars)
        E_optim = tf.train.AdamOptimizer(G_lr).minimize(VAE_loss, var_list=E_vars)
        # E_optim = tf.train.RMSPropOptimizer(lr).minimize(VAE_loss, var_list=E_vars)


    # open session and initialize all variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        logger = tf.summary.FileWriter('./graphs', sess.graph)
        merged = tf.summary.merge_all()


        if os.path.isdir(root) is False:
            os.mkdir(root)

        model_path = './network_test_chair/'
        if os.path.isdir(model_path) is False:
            os.mkdir(model_path)
        saver = tf.train.Saver()

        # training-loop
        num = 0
        for batch in range(20000):
            #x_im, x_3d = dataset.get_batch(batch_size)
            for nch in range(Nchannels):
                chn_data = np.random.rand



            x_im =  np.random.rand(batch_size,Input_Cube_Size,Input_Cube_Size,In,1)
            x_3d =  np.random.rand(batch_size,32,32,32,1)
            for _ in range(4):
                sess.run(D_optim, feed_dict={x_image: x_im, x_3D: x_3d, keep_prob: 0.8, isTrain: True})
                sess.run(clip)
            loss_d_, loss_g_, _VAE_loss, _KL_divergence, _reconstruction_loss, summary, _, _, _ = \
                sess.run([D_loss, G_loss, VAE_loss, mean_KL, mean_recon, merged, D_optim, G_optim, E_optim],
                         {x_image: x_im, x_3D: x_3d, keep_prob: 0.8, isTrain: True})
            sess.run(clip)
            if batch % 1 == 0:
                print("batch:", batch)
                print("D Loss:", loss_d_)
                print("G Loss:", loss_g_)
                print("VAE loss:", _VAE_loss)
                print("KL divergence:", _KL_divergence)
                print("reconstruction_loss:", _reconstruction_loss)
                print("###########")
                if batch % 500 == 0:
                    G = sess.run(G_z, feed_dict={x_image: x_im, x_3D: x_3d, keep_prob: 1, isTrain: False})
                    np.save(root + str(batch) + ".npy", G)
                    if batch % 1000 == 0:
                        saver.save(sess, model_path + str(batch) + ".ckpt")

        #sess.close()
