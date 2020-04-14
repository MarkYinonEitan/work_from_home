import os, time, sys, itertools
import numpy as np
import matplotlib
import tensorflow as tf

from gan_ops import batch_normal, de_conv3, conv3d, fully_connect



import dbloader
from dbloader import VOX_SIZE, RESOLUTION, N_SAMPLS_FOR_1V3
from dbloader import N_CHANNELS, BATCH_SIZE, MAP_BOX_SIZE, VX_BOX_SIZE

matplotlib.use('Agg')
import matplotlib.pyplot as plt

D_lr = 5e-5
G_lr = 1e-4
alpha_1 = 50
alpha_2 = 0.05
d_scale_factor=0.000000000025
g_scale_factor=0.000000000075


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

        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE ,1])
        mp_fake = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE ,1])

        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VX_BOX_SIZE, VX_BOX_SIZE, VX_BOX_SIZE ,N_CHANNELS])
        vx_fake = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VX_BOX_SIZE, VX_BOX_SIZE, VX_BOX_SIZE , N_CHANNELS])

        # inputs
        self.mp_real = mp_real
        self.mp_fake = mp_fake

        self.vx_real = vx_real
        self.vx_fake = vx_fake

        self.mp_rand = tf.random_normal(shape=self.mp_real.get_shape(),mean = dbloader.MEAN,stddev = dbloader.SIGMA)

        self.D_real = discriminate(self.mp_real,self.vx_real, reuse=False)
        self.D_rand = discriminate(self.mp_rand, self.vx_real, reuse=True)
        self.D_fake = discriminate(self.mp_fake,self.vx_real, reuse=True)
        #networks
        self.D_real_c = tf.clip_by_value(self.D_real,0.05,0.95)
        self.D_fake_c = tf.clip_by_value(self.D_fake,0.05,0.95)
        self.D_rand_c = tf.clip_by_value(self.D_rand,0.05,0.95)
        self.D_loss_real = -tf.reduce_mean(tf.log(self.D_real_c))
        self.D_loss_fake = -tf.reduce_mean(tf.log(1-self.D_fake_c))
        self.D_loss_rand = -tf.reduce_mean(tf.log(1-self.D_rand_c))


        self.D_loss = self.D_loss_real + self.D_loss_fake + self.D_loss_rand

        t_vars = tf.trainable_variables()

        self.log_vars.append(("discriminator_loss", self.D_loss))

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.saver = tf.train.Saver()

        #trainers
        self.trainer_D = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        self.gradients_D = self.trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        self.opti_D = self.trainer_D.apply_gradients(self.gradients_D)

class VAE_GAN1():
    def __init__(self):

        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE ,1])
        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VX_BOX_SIZE, VX_BOX_SIZE, VX_BOX_SIZE ,N_CHANNELS])

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
        self.kl_loss = tf.reduce_mean(KL_loss(self.z_mean, self.z_sigm)/(BATCH_SIZE*125))

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
        self.D_real = discriminate(self.mp_real,self.vx_real, reuse=False)
        self.D_rand = discriminate(self.mp_rand, self.vx_real, reuse=True)
        self.D_fake = discriminate(self.mp_fake,self.vx_real, reuse=True)


        #For encode
        self.encode_loss = tf.reduce_mean(self.kl_loss) #- self.LL_loss / (256*100.0)


        self.D_real_c = tf.clip_by_value(self.D_real,0.01,0.99)
        self.D_fake_c = tf.clip_by_value(self.D_fake,0.01,0.99)
        self.D_loss_real = -tf.reduce_mean(tf.log(self.D_real_c))
        self.D_loss_fake = -tf.reduce_mean(tf.log(1-self.D_fake_c))
        self.L_DISC = self.D_loss_real +self.D_loss_fake
        self.L_total = -self.D_loss_fake +alpha_1 *self.kl_loss+alpha_2*self.reconstr_loss

        self.Disc_Acc = (tf.math.reduce_sum(tf.math.sign(tf.math.sign(self.D_real-0.5) + 1) +tf.math.sign(tf.math.sign(0.5-self.D_fake)+1)))/2/BATCH_SIZE

        self.D_loss = self.L_DISC
        self.VAE_GAN_loss = self.L_total

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]


        #for D
        trainer_D = tf.train.AdamOptimizer()
        gradients_D = trainer_D.compute_gradients(self.D_loss, var_list=self.d_vars)
        self.opti_D = trainer_D.apply_gradients(gradients_D)

        #for G
        trainer_EC = tf.train.AdamOptimizer()
        gradients_EC = trainer_EC.compute_gradients(self.VAE_GAN_loss, var_list=self.g_vars+self.e_vars)
        self.opti_EC = trainer_EC.apply_gradients(gradients_EC)

#class NNS():
#    @stati
class VAE_V2():
    def __init__(self):

        mp_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE, MAP_BOX_SIZE ,1])
        vx_real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VX_BOX_SIZE, VX_BOX_SIZE, VX_BOX_SIZE ,N_CHANNELS])

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
        self.kl_loss = tf.reduce_mean(KL_loss(self.z_mean, self.z_sigm)/(BATCH_SIZE*125))

        'DECODER (GENERATOR)'
        #inputs
        self.ep = tf.random_normal(shape=self.z_mean.get_shape(),stddev=0.1)
        self.zp = tf.random_normal(shape=self.z_mean.get_shape())
        self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)
        #generate
        self.mp_fake = Generate(self.z_x, reuse=False)
        self.mp_rand = Generate(self.zp, reuse=True)
        self.reconstr_loss = tf.losses.mean_squared_error(self.mp_real, self.mp_fake)


        #For encode
        self.encode_loss = tf.reduce_mean(self.kl_loss) #- self.LL_loss / (256*100.0)
        t_vars = tf.trainable_variables()

        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.g_vars = [var for var in t_vars if 'gen_' in var.name]


        self.generator_loss = self.reconstr_loss + self.encode_loss   # average over batch

        self.log_vars.append(("e_loss", self.encode_loss))
        self.log_vars.append(("g_loss", self.generator_loss))
        self.log_vars.append(("reconstr_loss", self.reconstr_loss))
        self.saver = tf.train.Saver()


        #trainers
        #VAE
        self.trainer_VAE = tf.train.RMSPropOptimizer(learning_rate=self.new_learning_rate)
        self.gradients_VAE = self.trainer_VAE.compute_gradients(self.generator_loss, var_list=self.e_vars+self.g_vars)
        self.opti_VAE = self.trainer_VAE.apply_gradients(self.gradients_VAE)

        self.dbg1 = self.encode_loss
        self.dbg2 = self.reconstr_loss


def KL_loss( mu, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

def Generate( z_var, reuse=False):

    # 1st hidden layer
    #conv1 = tf.layers.conv3d_transpose(x, 512, [2, 2, 2], strides=(2, 2, 2), padding='valid', use_bias=False,
    #conv2 = tf.layers.conv3d_transpose(lelu1, 256, [2, 2, 2], strides=(1, 1, 1), padding='valid', use_bias=False,
    #conv3 = tf.layers.conv3d_transpose(lelu2, 128, [3, 3, 3], strides=(1, 1, 1), padding='valid', use_bias=False,
    #conv4 = tf.layers.conv3d_transpose(lelu3, 1, [1, 1, 1], strides=(1, 1, 1), padding='valid', use_bias=False,

    with tf.variable_scope('generator') as scope:

        if reuse == True:
            scope.reuse_variables()

        #d1 = tf.nn.elu(batch_normal(fully_connect(z_var , output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))

        d2 = tf.nn.elu(batch_normal(de_conv3(z_var , output_shape=[BATCH_SIZE, 5, 5, 5, 125],  k_dwh=[2,2,2],  d_dwh=[1,1,1],pad='SAME', name='gen_deconv2'), scope='gen_bn2', reuse=reuse))

        d3 = tf.nn.elu(batch_normal(de_conv3(d2, output_shape=[BATCH_SIZE, 5, 5, 5,126],  k_dwh=[2,2,2],  d_dwh=[1,1,1],pad = 'SAME',name='gen_deconv3'), scope='gen_bn3', reuse=reuse))

        d4 = tf.nn.elu(batch_normal(de_conv3(d3, output_shape=[BATCH_SIZE, 5, 5, 5,127], k_dwh=[2,2,2],  d_dwh=[1,1,1],pad = 'SAME', name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
        d5 = de_conv3(d4, output_shape=[BATCH_SIZE, 5, 5, 5,1],  k_dwh=[2,2,2], pad = 'SAME', d_dwh=[1,1,1],name='gen_deconv5')

        return d5 #tf.nn.elu(d5)

def Encode(x):

    with tf.variable_scope('encode', reuse = tf.AUTO_REUSE) as scope:

        conv1 = tf.nn.elu(batch_normal(conv3d(x, output_dim=32, k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1),  pad='VALID', name='e_c1'), scope='e_bn1'))
        conv2 = tf.nn.elu(batch_normal(conv3d(conv1, output_dim=128, k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1), pad='VALID',name='e_c2'), scope='e_bn2'))
        conv3_1 = tf.nn.elu(batch_normal(conv3d(conv2 , output_dim=128,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='e_c31'), scope='e_bn31'))
        conv3_2 = tf.nn.elu(batch_normal(conv3d(conv2 , output_dim=128,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='e_c32'), scope='e_bn32'))

        z_mean = conv3_1
        z_sigma = conv3_2

        return z_mean, z_sigma

def discriminate( mp, vx, reuse=False):
    paddings =  tf.constant([[0,0],
                             [(VX_BOX_SIZE-MAP_BOX_SIZE)//2,(VX_BOX_SIZE-MAP_BOX_SIZE)//2],\
                             [(VX_BOX_SIZE-MAP_BOX_SIZE)//2,(VX_BOX_SIZE-MAP_BOX_SIZE)//2],\
                             [(VX_BOX_SIZE-MAP_BOX_SIZE)//2,(VX_BOX_SIZE-MAP_BOX_SIZE)//2],\
                             [0,0]])
    mp_pad = tf.pad(mp,paddings,mode='CONSTANT')
    x_var= tf.concat([vx, mp_pad], 4)

    with tf.variable_scope("discriminator") as scope:

        if reuse:
            scope.reuse_variables()

        conv1 = tf.nn.elu(conv3d(x_var, output_dim=32, k_dwh=[1, 1, 1],  d_dwh=(1, 1, 1), pad='SAME', name='dis_conv1'))
        conv2= tf.nn.elu(batch_normal(conv3d(conv1, output_dim=32,k_dwh=[3, 3, 3],  d_dwh=(1, 1, 1), pad='VALID', name='dis_conv2'), scope='dis_bn1', reuse=reuse))
        conv3= tf.nn.elu(batch_normal(conv3d(conv2, output_dim=32,k_dwh=[2, 2, 2],  d_dwh=(1, 1, 1), pad='VALID', name='dis_conv3'), scope='dis_bn2', reuse=reuse))
        conv4 = conv3d(conv3, output_dim=128,k_dwh=[2, 2, 2],  d_dwh=(2, 2, 2),pad='VALID', name='dis_conv4')
        middle_conv = conv4
        conv4= tf.nn.elu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
        conv4= tf.reshape(conv4, [BATCH_SIZE, -1])

        fl = tf.nn.elu(batch_normal(fully_connect(conv4, output_size=32, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
        output = fully_connect(fl , output_size=1, scope='dis_fully2')

        dis_out = tf.sigmoid(output)

        return dis_out

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)
