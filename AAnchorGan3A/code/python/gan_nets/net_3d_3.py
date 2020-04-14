import os, time, sys, itertools
import numpy as np
import matplotlib
import tensorflow as tf

from ops import batch_normal, de_conv3, conv3d, fully_connect, lrelu


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

D_lr = 5e-5
G_lr = 1e-4
alpha_1 = 5
alpha_2 = 5e-4
d_scale_factor=0.25
g_scale_factor=0.75



#class NNS():
#    @stati
class VAE_GAN1():
    def __init__(self):

        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_OUT, NBOX_OUT, NBOX_OUT ,1])
        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NBOX_IN, NBOX_IN, NBOX_IN ,N_CHANNELS])

        self.vx_real = vx_real
        self.mp_real = mp_real

        self.learn_rate_init = 0.03

        #Learning Rate
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.new_learning_rate = tf.train.exponential_decay(self.learn_rate_init, global_step=self.global_step, decay_steps=10000,
                                                   decay_rate=0.98)

        self.log_vars = []



        'ENCODER'
        self.z_mean, self.z_sigm = Encode(self.vx_real)
        #KL loss
        self.kl_loss = KL_loss(self.z_mean, self.z_sigm)/(BATCH_SIZE*125)

        'DECODER (GENERATOR)'
        #inputs
        self.ep = tf.random_normal(shape=self.z_mean.get_shape(),stddev=0.1)
        self.zp = tf.random_normal(shape=self.z_mean.get_shape())
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        #generate
        self.mp_fake = Generate(self.z_x, reuse=False)
        self.mp_rand = Generate(self.zp, reuse=True)
        self.reconstr_loss = tf.losses.mean_squared_error(self.mp_real, self.mp_fake)

        'DISCRIMINATOR'
        self.l_x_real,  self.D_real_logits = discriminate(self.mp_real,self.vx_real, reuse=False)
        _, self.D_rand_logits = discriminate(self.mp_rand, self.vx_real, reuse=True)
        self.l_x_fake, self.D_fake_logits = discriminate(self.mp_fake,self.vx_real, reuse=True)

        # D loss
        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_fake_logits), logits=self.D_fake_logits))
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_real_logits) - d_scale_factor, logits=self.D_real_logits))
        self.D_rand_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_rand_logits), logits=self.D_rand_logits))

        # G loss
        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_fake_logits) - g_scale_factor, logits=self.D_fake_logits))
        self.G_rand_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_rand_logits) - g_scale_factor, logits=self.D_rand_logits))

        self.D_loss = self.D_fake_loss + self.D_real_loss + self.D_rand_loss

        # preceptual loss(feature loss)
        self.LL_loss = tf.reduce_mean(tf.reduce_sum(NLLNormal(self.l_x_real, self.l_x_fake), [1,2,3,4]))

        #For encode
        self.encode_loss = self.kl_loss - self.LL_loss / (256*100.0)

        #for Gen
        self.G_loss = self.G_fake_loss + self.G_rand_loss + self.reconstr_loss- 1e-6*self.LL_loss

        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("LL_loss", self.LL_loss))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        #for D
        trainer_D = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        self.opti_D = trainer_D.apply_gradients(gradients_D)

        #for G
        trainer_G = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        gradients_G = trainer_G.compute_gradients(self.G_loss, var_list=self.g_vars)
        self.opti_G = trainer_G.apply_gradients(gradients_G)

        #for E
        trainer_E = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        gradients_E = trainer_E.compute_gradients(self.encode_loss, var_list=self.e_vars)
        self.opti_E = trainer_E.apply_gradients(gradients_E)

def NLLNormal( pred, target):

    c = -0.5 * tf.log(2 * np.pi)
    multiplier = 1.0 / (2.0 * 1)
    tmp = tf.square(pred - target)
    tmp *= -multiplier
    tmp += c

    return tmp

def KL_loss( mu, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

def Generate( z_var, reuse=False):

    # 1st hidden layer
    #conv1 = tf.layers.conv3d_transpose(x, 512, [2, 2, 2], strides=(2, 2, 2), padding='valid', use_bias=False,
    #conv2 = tf.layers.conv3d_transpose(lrelu1, 256, [2, 2, 2], strides=(1, 1, 1), padding='valid', use_bias=False,
    #conv3 = tf.layers.conv3d_transpose(lrelu2, 128, [3, 3, 3], strides=(1, 1, 1), padding='valid', use_bias=False,
    #conv4 = tf.layers.conv3d_transpose(lrelu3, 1, [1, 1, 1], strides=(1, 1, 1), padding='valid', use_bias=False,

    with tf.variable_scope('generator') as scope:

        if reuse == True:
            scope.reuse_variables()

        #d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))

        d2 = tf.nn.relu(batch_normal(de_conv3(z_var , output_shape=[BATCH_SIZE, 5, 5, 5, 125],  k_dwh=[2,2,2],  d_dwh=[1,1,1],pad='SAME', name='gen_deconv2'), scope='gen_bn2', reuse=reuse))

        d3 = tf.nn.relu(batch_normal(de_conv3(d2, output_shape=[BATCH_SIZE, 5, 5, 5,126],  k_dwh=[2,2,2],  d_dwh=[1,1,1],pad = 'SAME',name='gen_deconv3'), scope='gen_bn3', reuse=reuse))

        d4 = tf.nn.relu(batch_normal(de_conv3(d3, output_shape=[BATCH_SIZE, 5, 5, 5,127], k_dwh=[2,2,2],  d_dwh=[1,1,1],pad = 'SAME', name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
        d5 = de_conv3(d4, output_shape=[BATCH_SIZE, 5, 5, 5,1],  k_dwh=[2,2,2], pad = 'SAME', d_dwh=[1,1,1],name='gen_deconv5')

        return tf.nn.tanh(d5)

def Encode(x):

    with tf.variable_scope('encode', reuse = tf.AUTO_REUSE) as scope:

        conv1 = tf.nn.relu(batch_normal(conv3d(x, output_dim=32, k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1),  pad='VALID', name='e_c1'), scope='e_bn1'))
        conv2 = tf.nn.relu(batch_normal(conv3d(conv1, output_dim=128, k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1), pad='VALID',name='e_c2'), scope='e_bn2'))
        conv3_1 = tf.nn.relu(batch_normal(conv3d(conv2 , output_dim=128,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='e_c31'), scope='e_bn31'))
        conv3_2 = tf.nn.relu(batch_normal(conv3d(conv2 , output_dim=128,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='e_c32'), scope='e_bn32'))

        z_mean = conv3_1
        z_sigma = conv3_2

        return z_mean, z_sigma

def discriminate( mp, vx, reuse=False):
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
