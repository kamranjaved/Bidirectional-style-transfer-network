import numpy as np
import time
import os
import tensorflow as tf
import model
import input_data
import ADAIN
from utils import *


# GAN
class GAN(object):
    def __init__(self, config):
        self.med_domain = True
        # input batch
        self.input_image_size = config.img_size
        self.padding_size = 272
        self.crop_size = 256  # None or same with image_size, if you don't want to crop
        self.random_crop = True
        self.img_channels = 3   # original is 1
        self.tr_inp_dir = config.tr_dir  # "../data/matching/viewed_tmp"
        self.ts_inp_dir = config.ts_dir  # "../data/matching/viewed_tmp"
        self.tr_txt = config.tr_dir+'/'+config.tr_list  # "../data/matching/viewed_tmp/tr_list.txt"
        self.ts_txt = config.ts_dir+'/'+config.ts_list  # "../data/matching/viewed_tmp/ts_list.txt"
        self.batch_size = config.batch_size  # 8
        self.ts_batch_size = config.ts_batch_size  #1
        self.flip = True
        self.num_identity = None #config.num_identity  # num of training data
        # network architecture
        self.generator_model = config.g_model  # 'alex'  # Select model
        self.use_enc_dec = config.use_enc_dec
        self.g_encoder_model = config.g_enc_model
        self.g_decoder_model = config.g_dec_model
        self.med_channels = config.med_channels
        self.discriminator_model = config.d_model  # 'alex'  # Select model
        self.share = 9999  # weight sharing is started at
        self.normG = 'batch_instance' #'instance'
        self.normD = 'batch_instance' #'batch'
        self.gan_loss = 'log'
        self.med_domain = True
        self.discriminator_method = 'col-cgan'
        self.similarity_loss = config.similarity_loss
        self.similarity_lambda = config.similarity_lambda
        self.similarity_loss_med = config.similarity_loss_med
        self.med_lambda = config.med_lambda
        self.med_step = config.med_step
        self.similarity_med = True
        self.cycle = False
        self.weight_decay = None  # 0.0005
        self.se_block = False
        self.share_g1 = False
        self.share_g2 = False
        if config.d_p2s == 'True':
            self.add_discriminators_p2s = True
        else:
            self.add_discriminators_p2s = False
        if config.d_s2p == 'True':
            self.add_discriminators_s2p = True
        else:
            self.add_discriminators_s2p = False
        self.med_d_lambda = config.med_d_lambda
        # trainer
        self.max_epoch = config.max_epoch  # 2000
        self.g_learning_rate = config.g_lr
        self.d_learning_rate = config.d_lr
        self.beta1 = 0.5
        self.beta2 = 0.999
        # learning rate decay
        self.lr_decay = None  # 0.1
        self.lr_step = 800
        self.step = tf.Variable(0, trainable=False, name='global_step')
        # records
        self.log_dir = config.log_dir  # "record"
        self.summary_step = 50  # write tensorboard summary
        self.print_epoch = config.print_epoch  # print losses
        self.save_epoch = config.save_epoch  # save model
        self.display_epoch = 1  #250
        if config.use_enc_dec:
            self.display_list = ['inp_photo', 'gen_photo', 'gen_sketch', 'inp_sketch']
        else:
            self.display_list = ['inp_photo', 'gen_photo', 'med_s2p', 'med_p2s', 'gen_sketch', 'inp_sketch']
        # to continue training, you must use same network architecture
        self.continue_training = config.continue_tr  # False
        self.load_dir = self.log_dir + "/gan_ckpt"
        self.load_epoch = config.load_epoch  # 500
        # Select GPU
        self.gpu_num = config.gpu_num

    def config_network(self, model_key, norm='instance', output_channels=1, output_activation='lrelu'):
        # select network model
        # layer specs = (type, out_channels, stride, ksize)
        print(model_key)
        if (model_key == 'pix2pix') or (model_key == 'p2p'):
            build_func = model.Unet
            config = {'layer_specs': [('conv', 64, 1, 4), ('conv', 128, 2, 4), ('conv', 256, 2, 4),
                                      ('conv', 512, 2, 4), ('conv', 512, 2, 4), ('conv', 512, 2, 4),
                                      ('conv', 512, 2, 4), ('conv', 512, 1, 4)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'col_gen') or (model_key == 'Col_Gen') or (model_key == 'res9'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('conv', 64, 1, 7), ('conv', 128, 2, 3), ('conv', 256, 2, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3),
                                      ('res', 256, 1, 3), ('dconv', 128, 2, 3), ('dconv', 64, 2, 3), ('conv', output_channels, 1, 7)],
                      'rectifier': 'relu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'res4'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('conv', 64, 1, 7), ('conv', 128, 2, 3), ('conv', 256, 2, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), 
                                      ('dconv', 128, 2, 3), ('dconv', 64, 2, 3), ('conv', output_channels, 1, 7)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'col_gen_enc') or (model_key == 'Col_Gen_enc'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('conv', 64, 1, 7), ('conv', 128, 2, 3), ('conv', 256, 2, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3)],
                      'rectifier': 'relu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': True, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'col_gen_dec') or (model_key == 'Col_Gen_dec'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3),
                                      ('dconv', 128, 2, 3), ('dconv', 64, 2, 3), ('conv', output_channels, 1, 7)],
                      'rectifier': 'relu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'col_gen_old'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('conv', 64, 1, 7), ('conv', 128, 2, 3), ('conv', 256, 2, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3),
                                      ('res', 256, 1, 3), ('dconv_o', 128, 2, 3), ('dconv_o', 64, 2, 3), ('conv', output_channels, 1, 7)],
                      'rectifier': 'relu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'res4_old'):
            build_func = model.CNN_Encoder
            config = {'layer_specs': [('conv', 64, 1, 7), ('conv', 128, 2, 3), ('conv', 256, 2, 3),
                                      ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), ('res', 256, 1, 3), 
                                      ('dconv_o', 128, 2, 3), ('dconv_o', 64, 2, 3), ('conv', output_channels, 1, 7)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        elif (model_key == 'patchgan') or (model_key == 'PatchGAN'):
            build_func = model.CNN_Encoder
            #  sigmoid at last layer
            config = {'layer_specs': [('conv', 64, 2, 4), ('conv', 128, 2, 4), ('conv', 256, 2, 4),
                                      ('conv', 512, 2, 4), ('conv', output_channels, 1, 4)],
                      'rectifier': 'lrelu', 'norm': norm, 'padding': 'SAME', 'use_bias': True,
                      'se-block': self.se_block, 'output_channels': output_channels,
                      'output_activation': output_activation}
        else:
            assert False, 'Config_network: Wrong model'

        return build_func, config

    def build_network(self, mode='train', test_gallery=False):
        # config generator
        if self.use_enc_dec:
            print('Use enc dec')
            encoder, self.g_enc_config = self.config_network(self.g_encoder_model, self.normG,
                                                             output_channels=self.med_channels, output_activation=None)
            decoder, self.g_dec_config = self.config_network(self.g_decoder_model, self.normG,
                                                             output_channels=self.img_channels, output_activation='tanh')
        else:
            generator, self.g_config = self.config_network(self.generator_model, self.normG, output_channels=self.img_channels,
                                                       output_activation='tanh')

        # photo to sketch
        if self.med_domain:
            if self.use_enc_dec:
                self.gen_med_p2s = encoder(self.photo_inp, self.g_enc_config, self.train_mode, name="Gen_p2s_A_",
                                           reuse=False, share_name='Gen_A_', share_reuse=False, share=self.share)
                if self.share_g1:
                    print('using shared weights')
                    self.gen_med_s2p = encoder(self.sketch_inp, self.g_enc_config, self.train_mode, name="Gen_p2s_A_",
                                               reuse=True, share_name='Gen_A_', share_reuse=True, share=self.share)
                else:
                    self.gen_med_s2p = encoder(self.sketch_inp, self.g_enc_config, self.train_mode, name="Gen_s2p_A_",
                                               reuse=False, share_name='Gen_A_', share_reuse=False, share=self.share)

                #self.gen_med_s2p = encoder(self.sketch_inp, self.g_enc_config, self.train_mode, name="Gen_s2p_A_",
                                           #reuse=False, share_name='Gen_A_', share_reuse=True, share=self.share)

                #self.gen_med_p2s = model.AdaIN(self.gen_med_p2s, self.gen_med_s2p) #, self.train_mode, name="Gen_p2s_A_",
                #self.gen_med_s2p = model.AdaIN(self.gen_med_s2p, self.gen_med_p2s) #, self.train_mode, name="Gen_p2s_A_",
#--------------------------
                #self.gen_med_p2s, self.content_p2s = model.AdaIN_update(self.gen_med_p2s, self.gen_med_s2p) #, self.train_mode, name="Gen_p2s_A_",
                #self.gen_med_s2p, self.content_s2p = model.AdaIN_update(self.gen_med_s2p, self.gen_med_p2s) #, self.train_mode, name="Gen_p2s_A_",

                #self.gen_med_p2s, self.content_p2s, self.style_p2s, self.gen_med_s2p, self.content_s2p, self.style_s2p = model.AdaIN_p2s_s2p_new(self.gen_med_p2s, self.gen_med_s2p) #
                #self.gen_med_p2s, self.content_p2s, self.gen_med_s2p, self.content_s2p = model.AdaIN_p2s_s2p(self.gen_med_p2s, self.gen_med_s2p) #
                self.gen_med_p2s, self.gen_med_p2s1, self.content_p2s, self.gen_med_s2p, self.gen_med_s2p1, self.content_s2p = model.AdaIN_p2s_s2p_new(self.gen_med_p2s, self.gen_med_s2p) #

#-------------------------
                if test_gallery:
                    self.gallery_med = encoder(self.gallery_inp, self.g_enc_config, self.train_mode, name="Gen_p2s_A_",
                                           reuse=True, share_name='Gen_A_', share_reuse=True, share=self.share)
                self.gen_sketch = decoder(self.gen_med_p2s, self.g_dec_config, self.train_mode, name="Gen_p2s_B_",
                                          reuse=False, share_name='Gen_B_', share_reuse=False, share=-self.share)
                self.gen_sketch1 = decoder(self.gen_med_p2s1, self.g_dec_config, self.train_mode, name="Gen_p2s_B_",
                                          reuse=True, share_name='Gen_B_', share_reuse=True, share=-self.share)
                if self.share_g2:
                    self.gen_photo = decoder(self.gen_med_s2p, self.g_dec_config, self.train_mode, name="Gen_p2s_B_",
                                          reuse=True, share_name='Gen_B_', share_reuse=True, share=-self.share)
                    self.gen_photo = decoder(self.gen_med_s2p1, self.g_dec_config, self.train_mode, name="Gen_p2s_B_",
                                          reuse=True, share_name='Gen_B_', share_reuse=True, share=-self.share)
                else:
                    self.gen_photo = decoder(self.gen_med_s2p, self.g_dec_config, self.train_mode, name="Gen_s2p_B_",
                                           reuse=False, share_name='Gen_B_', share_reuse=False, share=self.share)
                    self.gen_photo1 = decoder(self.gen_med_s2p1, self.g_dec_config, self.train_mode, name="Gen_s2p_B_",
                                           reuse=True, share_name='Gen_B_', share_reuse=True, share=self.share)
            else:
                self.gen_med_p2s = generator(self.photo_inp, self.g_config, self.train_mode, name="Gen_p2s_A_",
                                           reuse=False, share_name='Gen_A_', share_reuse=False, share=self.share)
                if test_gallery:
                    self.gallery_med = generator(self.gallery_inp, self.g_config, self.train_mode, name="Gen_p2s_A_",
                                           reuse=True, share_name='Gen_A_', share_reuse=True, share=self.share)
                self.gen_sketch = generator(self.gen_med_p2s, self.g_config, self.train_mode, name="Gen_p2s_B_",
                                           reuse=False, share_name='Gen_B_', share_reuse=False, share=self.share)
                self.gen_med_s2p = generator(self.sketch_inp, self.g_config, self.train_mode, name="Gen_s2p_A_",
                                           reuse=False, share_name='Gen_A_', share_reuse=True, share=self.share)
                self.gen_photo = generator(self.gen_med_s2p, self.g_config, self.train_mode, name="Gen_s2p_B_",
                                           reuse=False, share_name='Gen_B_', share_reuse=True, share=self.share)
        else:
            self.gen_sketch = generator(self.photo_inp, self.g_config, self.train_mode, name="Gen_p2s_", reuse=False)
            self.gen_photo = generator(self.sketch_inp, self.g_config, self.train_mode, name="Gen_s2p_", reuse=False)

        if (mode == 'train_gan') or (mode == 'train_with_matching'):
            # discriminator inputs
            if (self.discriminator_method == 'col-cgan') or (self.discriminator_method == 'Col-cGAN'):
                if self.use_enc_dec:
                    print('use_enc_dec: no col-cgan')
                    self.genAB_sk = tf.concat([self.photo_inp, self.gen_sketch], axis=3)
                    self.genAB_sk1 = tf.concat([self.photo_inp, self.gen_sketch1], axis=3)
                    self.genAB_ph = tf.concat([self.gen_photo, self.sketch_inp], axis=3)
                    self.genAB_ph1 = tf.concat([self.gen_photo1, self.sketch_inp], axis=3)
                    self.realAB_p2s = tf.concat([self.photo_inp, self.sketch_inp], axis=3)
                    self.realAB_s2p = tf.concat([self.photo_inp, self.sketch_inp], axis=3)
                else:
                    self.genAB_sk = tf.concat([self.photo_inp, self.gen_med_p2s, self.gen_sketch], axis=3)
                    self.genAB_sk1 = tf.concat([self.photo_inp, self.gen_med_p2s1, self.gen_sketch1], axis=3)
                    self.genAB_ph = tf.concat([self.gen_photo, self.gen_med_s2p, self.sketch_inp], axis=3)
                    self.genAB_ph1 = tf.concat([self.gen_photo1, self.gen_med_s2p1, self.sketch_inp], axis=3)
                    self.realAB_p2s = tf.concat([self.photo_inp, self.gen_med_p2s, self.sketch_inp], axis=3)
                    self.realAB_s2p = tf.concat([self.photo_inp, self.gen_med_s2p, self.sketch_inp], axis=3)
            elif (self.discriminator_method == 'cgan') or (self.discriminator_method == 'pix2pix'):
                self.genAB_sk = tf.concat([self.photo_inp, self.gen_sketch], axis=3)
                self.genAB_sk1 = tf.concat([self.photo_inp, self.gen_sketch1], axis=3)
                self.genAB_ph = tf.concat([self.gen_photo, self.sketch_inp], axis=3)
                self.genAB_ph1 = tf.concat([self.gen_photo1, self.sketch_inp], axis=3)
                self.realAB_p2s = tf.concat([self.photo_inp, self.sketch_inp], axis=3)
                self.realAB_s2p = tf.concat([self.photo_inp, self.sketch_inp], axis=3)
            else:
                assert False, 'discriminator method error'

            # config discriminator
            discriminator, self.d_config = self.config_network(self.discriminator_model, self.normD, output_channels=1,
                                                               output_activation='sigmoid')
            # build discriminators
            self.dreal_p2s = discriminator(self.realAB_p2s, self.d_config, self.train_mode, name="Dis_p2s_", reuse=False)
            self.dfake_p2s = discriminator(self.genAB_sk, self.d_config, self.train_mode, name="Dis_p2s_", reuse=True)
            self.dfake_p2s1 = discriminator(self.genAB_sk1, self.d_config, self.train_mode, name="Dis_p2s_", reuse=True)
            self.dreal_s2p = discriminator(self.realAB_s2p, self.d_config, self.train_mode, name="Dis_s2p_", reuse=False)
            self.dfake_s2p = discriminator(self.genAB_ph, self.d_config, self.train_mode, name="Dis_s2p_", reuse=True)
            self.dfake_s2p1 = discriminator(self.genAB_ph1, self.d_config, self.train_mode, name="Dis_s2p_", reuse=True)

            # adversarial loss
            self.adv_loss_p2s, self.d_loss_p2s = model.GAN_loss_bin(self.dfake_p2s, self.dreal_p2s, self.gan_loss)
            self.adv_loss_p2s1, self.d_loss_p2s1 = model.GAN_loss_bin(self.dfake_p2s1, self.dreal_p2s, self.gan_loss)

            self.adv_loss_s2p, self.d_loss_s2p = model.GAN_loss_bin(self.dfake_s2p, self.dreal_s2p, self.gan_loss)
            self.adv_loss_s2p1, self.d_loss_s2p1 = model.GAN_loss_bin(self.dfake_s2p1, self.dreal_s2p, self.gan_loss)

            #self.adv_loss = self.adv_loss_p2s + self.adv_loss_s2p
            self.adv_loss = self.adv_loss_p2s + self.adv_loss_p2s1 + self.adv_loss_s2p + self.adv_loss_s2p1
            self.g_loss = self.adv_loss
            #self.d_loss = self.d_loss_p2s + self.d_loss_s2p
            self.d_loss = self.d_loss_p2s + self.d_loss_s2p + self.d_loss_p2s1 + self.d_loss_s2p1

            if self.add_discriminators_p2s:
                print('discriminators_at_intermediate_domain_p2s')
                self.dreal_m_p2s = discriminator(self.gen_med_s2p, self.d_config, self.train_mode, name="Dis_med_p2s_", reuse=False)
                self.dfake_m_p2s = discriminator(self.gen_med_p2s, self.d_config, self.train_mode, name="Dis_med_p2s_", reuse=True)
                self.adv_loss_m_p2s, self.d_loss_m_p2s = model.GAN_loss_bin(self.dfake_m_p2s, self.dreal_m_p2s, self.gan_loss)
                self.g_loss = self.g_loss + (self.med_d_lambda * self.adv_loss_m_p2s)
                self.d_loss = self.d_loss + (self.med_d_lambda * self.d_loss_m_p2s)
            if self.add_discriminators_s2p:
                print('discriminators_at_intermediate_domain_s2p')
                self.dreal_m_s2p = discriminator(self.gen_med_p2s, self.d_config, self.train_mode, name="Dis_med_s2p_", reuse=False)
                self.dfake_m_s2p = discriminator(self.gen_med_s2p, self.d_config, self.train_mode, name="Dis_med_s2p_", reuse=True)
                self.adv_loss_m_s2p, self.d_loss_m_s2p = model.GAN_loss_bin(self.dfake_m_s2p, self.dreal_m_s2p, self.gan_loss)
                self.g_loss = self.g_loss + (self.med_d_lambda * self.adv_loss_m_s2p)
                self.d_loss = self.d_loss + (self.med_d_lambda * self.d_loss_m_s2p)

            # # for wgan-gp
            # if self.gan_loss == 'wgan-gp':
            #     def interpolate(a, b):
            #         shape = tf.stack([tf.shape(a)[0], 1, 1, 1])
            #         alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            #         inter = a + alpha * (b - a)
            #         inter.set_shape(a.get_shape().as_list())
            #         return inter
            #     print('gredient penalty')
            #     x = interpolate(self.realAB, self.genAB)
            #     pred = discriminator(x, self.n_layer_d, self.d_config, self.train_mode, name="Dis_", reuse=True)
            #     if len(pred.get_shape().as_list()) > 2:
            #         pred = tf.reduce_mean(pred, axis=[1, 2])
            #     gradients = tf.gradients(pred, x)[0]
            #     slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            #     self.gp = tf.reduce_mean((slopes - 1.) ** 2)
            #     self.d_loss = self.d_loss + self.gp_lambda * self.gp
            #     gp_sum = tf.summary.scalar("GP", self.gp)

            # generate cycle
            # if self.cycle:
            #     print('Cycle')
            #     # Cycle for s2p (sketch to photo to sketch)
            #     if self.med_domain:
            #         self.cycle_med_s2p2s = generator(self.gen_photo, self.g_config, self.train_mode, name="Gen_p2s_A_",
            #                                          reuse=True)
            #         self.cycle_sketch = generator(self.cycle_med_s2p2s, self.g_config, self.train_mode, name="Gen_p2s_B_",
            #                                       reuse=True)
            #     else:
            #         self.cycle_sketch = generator(self.gen_photo, self.g_config, self.train_mode, name="Gen_p2s_",
            #                                       reuse=True)
            #     # tf.summary.image('cycle_sketch', self.cycle_sketch, max_outputs=3)
            #
            #     # Cycle for p2s (photo to sketch to photo)
            #     if self.med_domain:
            #         self.cycle_med_p2s2p = generator(self.gen_sketch, self.g_config, self.train_mode, name="Gen_s2p_A_",
            #                                          reuse=True)
            #         self.cycle_photo = generator(self.gen_med_s2p, self.g_config, self.train_mode, name="Gen_s2p_B_",
            #                                      reuse=True)
            #     else:
            #         self.cycle_photo = generator(self.gen_sketch, self.g_config, self.train_mode, name="Gen_s2p_",
            #                                      reuse=True)
            #     # tf.summary.image('cycle_photo', self.cycle_photo, max_outputs=3)
            #
            #     # Cycle_consistency loss
            #     self.cycle_loss_photo = tf.reduce_mean(tf.abs(self.photo_inp - self.cycle_photo))
            #     self.cycle_loss_sketch = tf.reduce_mean(tf.abs(self.sketch_inp - self.cycle_sketch))
            #     self.g_loss = tf.add(self.g_loss, 10 * (self.cycle_loss_photo + self.cycle_loss_sketch))

            # ssim score
            # self.ssim_score = tf.image.ssim(self.gen, self.realB, max_val=2.0)
            # self.ssim_score_photo = tf.reduce_mean(tf_ssim(self.gen_photo, self.photo_inp))
            self.ssim_score_photo = tf.reduce_mean(tf.image.ssim(self.gen_photo, self.photo_inp, max_val=2.0))
            self.ssim_score_sketch = tf.reduce_mean(tf.image.ssim(self.gen_sketch, self.sketch_inp, max_val=2.0))
            self.average_ssim = (self.ssim_score_photo + self.ssim_score_sketch) / 2
            ssim_sum = tf.summary.scalar("SSIM", self.average_ssim)

            # similarity loss
            if self.similarity_loss is not None:
                def calc_similarity(imgA, imgB, metric, ssim_lambda=1.0):
                    if metric == 'L1':
                        # similarity = tf.reduce_mean(tf.abs(imgA - imgB))
                        similarity = tf.reduce_mean(tf.reduce_sum(tf.abs(imgA - imgB), axis=0))
                    elif metric == 'L2':
                        # similarity = tf.reduce_mean((imgA - imgB) ** 2)
                        similarity = tf.reduce_mean(tf.reduce_sum((imgA - imgB) ** 2, axis=0))
                    elif metric == 'ssim':
                        ssim_score = tf.reduce_mean(tf.image.ssim(imgA, imgB, max_val=2.0))
                        similarity = 1 - ssim_score
                    elif (metric == 'L1_ssim') or (metric == 'ssim_L1'):
                        ssim_score = tf.reduce_mean(tf.image.ssim(imgA, imgB, max_val=2.0))
                        # similarity = tf.reduce_mean(tf.abs(imgA - imgB)) + (ssim_lambda * (1 - ssim_score))
                        similarity = tf.reduce_mean(tf.reduce_sum(tf.abs(imgA - imgB), axis=0)) + (ssim_lambda * (1 - ssim_score))
                    else:
                        assert False, 'similarity loss error'
                    return similarity
                self.s_loss_photo = calc_similarity(self.gen_photo, self.photo_inp, self.similarity_loss)
                self.s_loss_sketch = calc_similarity(self.gen_sketch, self.sketch_inp, self.similarity_loss)

                self.s_loss_photo1 = calc_similarity(self.gen_photo1, self.photo_inp, self.similarity_loss)
                self.s_loss_sketch1 = calc_similarity(self.gen_sketch1, self.sketch_inp, self.similarity_loss)

                self.s_loss1 = tf.add(self.s_loss_photo,  self.s_loss_sketch)
                self.s_loss2 = tf.add(self.s_loss_photo1,  self.s_loss_sketch1)
                self.s_loss = tf.add(self.s_loss1, 0.5*self.s_loss2) 

                self.g_loss = tf.add(self.adv_loss, self.similarity_lambda * self.s_loss)
                s_loss_sum = tf.summary.scalar("Similarity_loss", self.s_loss)
                if self.similarity_med:
                    #self.s_loss_med = calc_similarity(self.gen_med_p2s, self.gen_med_s2p, self.similarity_loss_med)

#---------------------------
                    #self.s_loss_med_style = calc_similarity(self.style_p2s, self.style_s2p, self.similarity_loss_med)
                    self.s_loss_med = calc_similarity(self.content_p2s, self.content_s2p, self.similarity_loss_med)  # L1 loss for content vector of photo and sketch
#---------------------------
                    self.med_lambda_p = tf.placeholder(tf.float32, name='med_lambda')
                    s_loss_sum_med = tf.summary.scalar("Col_loss", self.s_loss_med)
                    #self.s_loss_med = tf.add(self.s_loss_med, 0.5*self.s_loss_med_style)
                    self.g_loss = tf.add(self.g_loss, self.med_lambda_p * self.s_loss_med)
                    s_loss_med_sum = tf.summary.scalar("col_loss", self.s_loss_med)

        return

    def build_optimizer(self):
        # variable list
        self.t_vars = tf.trainable_variables()
        self.Gen_vars = [var for var in self.t_vars if 'Gen_' in var.name]
        self.Dis_vars = [var for var in self.t_vars if 'Dis_' in var.name]

        # Weight decay
        if (self.weight_decay != None) and (self.weight_decay != False) and (self.weight_decay != 0):
            self.g_reg = tf.reduce_sum(
                [tf.nn.l2_loss(var, name='g_reg') for var in self.Gen_vars if 'weights' in var.name], name='Gen_reg')
            self.d_reg = tf.reduce_sum(
                [tf.nn.l2_loss(var, name='d_reg') for var in self.Dis_vars if 'weights' in var.name], name='Dis_reg')
            if self.weight_decay != 0:
                self.g_loss = tf.add(self.g_loss, self.weight_decay * self.g_reg, name='Gen_loss')
                self.d_loss = tf.add(self.d_loss, self.weight_decay * self.d_reg, name='Dis_loss')
        Gen_loss_sum = tf.summary.scalar("Generator_loss", self.g_loss)
        Dis_loss_sum = tf.summary.scalar("Discriminator_loss", self.d_loss)

        # learning rate
        if self.lr_decay == None:
            self.g_lr = self.g_learning_rate
            self.d_lr = self.d_learning_rate
        else:
            self.g_lr = tf.train.exponential_decay(self.g_learning_rate, self.step, self.lr_step, self.lr_decay,
                                                   staircase=self.lr_stair)
            self.d_lr = tf.train.exponential_decay(self.d_learning_rate, self.step, self.lr_step, self.lr_decay,
                                                   staircase=self.lr_stair)

        # Optimizer
        Gen_optimizer = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2, name='Gen_Adam')
        Dis_optimizer = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2, name='Dis_Adam')

        # Compute gradiants
        Gen_grad = Gen_optimizer.compute_gradients(self.g_loss, var_list=self.Gen_vars)
        Dis_grad = Dis_optimizer.compute_gradients(self.d_loss, var_list=self.Dis_vars)

        # Updates
        self.g_optim = Gen_optimizer.apply_gradients(Gen_grad, global_step=self.step)
        self.d_optim = Dis_optimizer.apply_gradients(Dis_grad)

        return

    def build_trainer(self, mode='train_gan'):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if self.crop_size is not None:
            self.image_size = self.crop_size
        else:
            self.image_size = self.input_image_size

        # placeholder
        self.train_mode = tf.placeholder(tf.bool, name='train_mode')

        # Input images
        # train batch
        self.tr_photo_inp, self.tr_sketch_inp, self.tr_photo_identity, self.tr_sketch_identity, self.tr_photo_name, self.tr_sketch_name, self.tr_photo_num, self.tr_sketch_num = \
            input_data.photo_sketch_batch_inputs(self.tr_inp_dir, self.tr_txt, self.tr_txt, self.num_identity, 1,
                                                 ['real_db'], self.batch_size, img_size=self.input_image_size,
                                                 name='tr_inp', photo_dim=self.img_channels, sketch_dim=self.img_channels,
                                                 flip=self.flip, crop_size=self.crop_size, padding_size=self.padding_size,
                                                 random_crop=self.random_crop, train_mode='train_gan',
                                                 concat_sketch_styles=True, log_dir=self.log_dir+'/tr_inputs.txt')
        # test batch
        self.ts_photo_inp, self.ts_sketch_inp, self.ts_photo_identity, self.ts_sketch_identity, self.ts_photo_name, self.ts_sketch_name, self.ts_photo_num, self.ts_sketch_num = \
            input_data.photo_sketch_batch_inputs(self.ts_inp_dir, self.ts_txt, self.ts_txt, self.num_identity, 1,
                                                 ['real_db'], self.ts_batch_size, img_size=self.input_image_size,
                                                 name='ts_inp', photo_dim=self.img_channels, sketch_dim=self.img_channels,
                                                 flip=False, crop_size=self.crop_size, padding_size=self.padding_size,
                                                 random_crop=False, train_mode='test_gan', concat_sketch_styles=True,
                                                 log_dir=self.log_dir+'/ts_inputs.txt')

#-------------------------------
        self.ts_sketch_inp = tf.manip.roll (self.ts_sketch_inp, shift=2, axis=0)

        # input condition train/test
        self.photo_inp = tf.cond(self.train_mode, lambda: self.tr_photo_inp, lambda: self.ts_photo_inp)
        #self.sketch_inp = tf.cond(self.train_mode, lambda: self.tr_sketch_inp, lambda: self.ts_sketch_inp)
#-----------
        self.sketch_inp = tf.cond(self.train_mode, lambda: self.tr_sketch_inp, lambda: self.ts_sketch_inp)
#--------
        self.photo_identity = tf.cond(self.train_mode, lambda: self.tr_photo_identity, lambda: self.ts_photo_identity)
        self.sketch_identity = tf.cond(self.train_mode, lambda: self.tr_sketch_identity, lambda: self.ts_sketch_identity)
        self.photo_name = tf.cond(self.train_mode, lambda: self.tr_photo_name, lambda: self.ts_photo_name)
        self.sketch_name = tf.cond(self.train_mode, lambda: self.tr_sketch_name, lambda: self.ts_sketch_name)

        if (mode == 'train_gan') or (mode == 'train_with_matching'):
            epoch = int(self.tr_photo_num / self.batch_size)
            print('Photo(train)')
            print(self.tr_photo_inp.get_shape())
            print(self.tr_photo_num)
            print('Sketch(train)')
            print(self.tr_sketch_inp.get_shape())
            print(self.tr_sketch_num)
        elif mode == 'test_gan':
            epoch = int(self.ts_photo_num / self.ts_batch_size)
        else:
            print('build_Trainer: Wrong mode 2')
            assert False

        # build network
        self.build_network(mode=mode)

        # print settings
        # modify later
        txtfile = open(self.log_dir + '/' + mode + '_config.txt', 'w')
        print(mode, file=txtfile)
        print('# Network settings=======================', file=txtfile)
        if self.use_enc_dec:
            print('generator_encoder_model: %s' % self.g_encoder_model, file=txtfile)
            print(self.g_enc_config, file=txtfile)
            print('generator_decoder_model: %s' % self.g_decoder_model, file=txtfile)
            print(self.g_dec_config, file=txtfile)
        else:
            print('generator_model: %s' % self.generator_model, file=txtfile)
            print(self.g_config, file=txtfile)
        print('discriminator_model: %s' % self.discriminator_model, file=txtfile)
        print(self.d_config, file=txtfile)
        print('use med_domain: %r' % self.med_domain, file=txtfile)
        print('discriminator method: %s' % self.discriminator_method, file=txtfile)
        print('similarity loss: %s' % self.similarity_loss, file=txtfile)
        print('similarity lambda: %f' % self.similarity_lambda, file=txtfile)
        print('similarity med: %s' % self.similarity_med, file=txtfile)
        if self.med_step is not None:
            print('Use meddle similarity after %d steps' % self.med_step, file=txtfile)
        print('med lambda: %f' % self.med_lambda, file=txtfile)
        print('med GAN lambda: %f' % self.med_d_lambda, file=txtfile)
        print('# Input settings=======================', file=txtfile)
        print('tr_batch_size: %d' % self.batch_size, file=txtfile)
        print('input_image_size: %d' % self.input_image_size, file=txtfile)
        print('padding_size: %d' % self.padding_size, file=txtfile)
        print('crop_size: %d' % self.crop_size, file=txtfile)
        print('random_crop: %r' % self.random_crop, file=txtfile)
        print('img_channels: %d' % self.img_channels, file=txtfile)
        print('flip: %r' % self.flip, file=txtfile)
        # print('num_identity: %d' % self.num_identity, file=txtfile)
        # print('style_list: %s' % self.style_list, file=txtfile)
        print('# Trainer settings=======================', file=txtfile)
        print('max_epoch: %d' % self.max_epoch, file=txtfile)
        print('generator learning_rate: %f' % self.g_learning_rate, file=txtfile)
        print('discriminator learning_rate: %f' % self.d_learning_rate, file=txtfile)
        print('beta1: %f' % self.beta1, file=txtfile)
        print('beta2: %f' % self.beta2, file=txtfile)
        if self.weight_decay is not None:
            print('weight_decay: %f' % self.weight_decay, file=txtfile)
        if self.lr_decay is not None:
            print('lr_decay: %f' % self.lr_decay, file=txtfile)
            print('lr_step: %f' % self.lr_step, file=txtfile)
        print("1 epoch: %d iterations" % epoch)
        txtfile.close()

        if (mode == 'train_gan'):
            self.build_optimizer()

        return

    def train_gan(self):
        epoch = int(self.tr_photo_num / self.batch_size)
        ts_epoch = int(self.ts_photo_num / self.ts_batch_size)
        print("1 epoch: %d iterations" % epoch)

        if not os.path.exists(self.log_dir + '/gan_ckpt'):
            os.mkdir(self.log_dir + '/gan_ckpt')
        if not os.path.exists(self.log_dir+'/gan_image'):
            os.mkdir(self.log_dir+'/gan_image')
        if not os.path.exists(self.log_dir + '/gan_summary'):
            os.mkdir(self.log_dir + '/gan_summary')
        if not os.path.exists(self.log_dir + '/gan_summary/' + str(self.max_epoch)):
            os.mkdir(self.log_dir + '/gan_summary/' + str(self.max_epoch))
        if self.continue_training:
            txtfile = open(self.log_dir + '/gan_log_' + str(self.max_epoch) + '.txt', 'w')
            txtfile_ts = open(self.log_dir+'/gan_log_ts-'+str(self.max_epoch)+'.txt', 'w')
        else:
            txtfile = open(self.log_dir + '/gan_log.txt', 'w')
            txtfile_ts = open(self.log_dir+'/gan_log_ts.txt', 'w')

        print("epoch\td_loss\tg_loss\tadv_loss\ts_loss\tssim", file=txtfile)
        print("epoch\td_loss\tg_loss\tadv_loss\ts_loss\tssim", file=txtfile_ts)

        # initializer
        init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(max_to_keep=None)
        # Summary
        self.merged_summary = tf.summary.merge_all()

        # feed_dict
        if (self.med_step is None) or (self.med_step <= 0):
            med_lambda = self.med_lambda
            self.med_step = -1
        else:
            med_lambda = 0
        feed_dict = {self.train_mode: True, self.med_lambda_p: med_lambda}
        feed_dict_ts = {self.train_mode: False, self.med_lambda_p: med_lambda}
        # fetch dict
        tr_dict = {'d_optim': self.d_optim, 'g_optim': self.g_optim, 'd_loss': self.d_loss, 'g_loss': self.g_loss,
                   'adv_loss': self.adv_loss, 's_loss': self.s_loss, 'average_ssim': self.average_ssim,
                   'photo_ssim_score': self.ssim_score_photo, 'sketch_ssim_score': self.ssim_score_sketch,
                   'name': self.photo_name, 'sk_name': self.sketch_name}
        ts_dict = {'inp_photo': self.photo_inp, 'inp_sketch': self.sketch_inp, 'gen_sketch': self.gen_sketch,
                   'gen_photo': self.gen_photo, 'med_p2s': self.gen_med_p2s, 'med_s2p': self.gen_med_s2p,
                   'name': self.photo_name, 'sk_name': self.sketch_name, 'd_loss': self.d_loss, 'g_loss': self.g_loss, 'adv_loss': self.adv_loss,
                   's_loss': self.s_loss,  'average_ssim': self.average_ssim, 'photo_ssim_score': self.ssim_score_photo,
                   'sketch_ssim_score': self.ssim_score_sketch}

        # ConfigProto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        # Training
        with tf.Session(config=config) as sess:
            print("Start session")
            summary_writer = tf.summary.FileWriter(self.log_dir + '/gan_summary/' + str(self.max_epoch), sess.graph)
            if self.continue_training:
                epoch_i = self.load_epoch
                self.saver.restore(sess, self.load_dir + "/gan-" + str(self.load_epoch))
                print("Model restored from epoch %d." % self.load_epoch)
            else:
                epoch_i = 0
                sess.run(init_op)
                print("Initialization done")

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Queue started")

            # Do training
            print("Start training")
            for i in range(epoch * epoch_i + 1, int(self.max_epoch * epoch) + 1):
                # epoch check
                if i % epoch == 1:
                    epoch_i = epoch_i + 1
                if i == self.med_step:
                    feed_dict.update({self.med_lambda_p: self.med_lambda})
                    feed_dict_ts.update({self.med_lambda_p: self.med_lambda})
                    print("Use med_similarity now")
                    print("Use med_similarity now", file=txtfile)

                # Update
                tr_result = sess.run(tr_dict, feed_dict=feed_dict)
                assert not np.isnan(tr_result['g_loss']), 'Model diverged with g_loss = NaN'
                assert not np.isnan(tr_result['d_loss']), 'Model diverged with d_loss = NaN'

                # records
                if i % self.summary_step == 0:
                    summary_str = sess.run(self.merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)

                # epoch check
                if i % epoch == 0:
                    # print
                    if epoch_i % self.print_epoch == 0:
                        cur_time = time.localtime(time.time())
                        print("%d.%d.%d %d:%d:%d||epoch %d. d_loss: %.5f g_loss: %.5f adv_loss: %.5f s_loss: %.5f ssim: %.5f"
                              % (cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour, cur_time.tm_min,
                                 cur_time.tm_sec, epoch_i, tr_result['d_loss'], tr_result['g_loss'], tr_result['adv_loss'],
                                 tr_result['s_loss'], tr_result['average_ssim']))
                        print("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f"
                              % (epoch_i, tr_result['d_loss'], tr_result['g_loss'], tr_result['adv_loss'],
                                 tr_result['s_loss'], tr_result['average_ssim']), file=txtfile)

                    # visualization
                    if epoch_i % self.display_epoch == 0:
                        vis_dir = self.log_dir + '/gan_image/ep' + str(epoch_i) + '_iter' + str(i)
                        print("============================")
                        print(self.log_dir)
                        g_loss = 0.
                        d_loss = 0.
                        adv_loss = 0.
                        s_loss = 0.
                        ssim = 0.
                        for j in range(ts_epoch):
                            ts_result = sess.run(ts_dict, feed_dict=feed_dict_ts)
                            # for k in range(len(ts_result['name'])):
                            #     assert ts_result['name'][k] == ts_result['sk_name'][k], 'Ts_inputs sequence error'
                            visualize_results(vis_dir, ts_result, self.display_list)
                            g_loss = g_loss + ts_result['g_loss']
                            d_loss = d_loss + ts_result['d_loss']
                            adv_loss = adv_loss + ts_result['adv_loss']
                            s_loss = s_loss + ts_result['s_loss']
                            ssim = ssim + ts_result['average_ssim']
                        print("%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f"
                              % (epoch_i, d_loss / ts_epoch, g_loss / ts_epoch, adv_loss / ts_epoch, s_loss / ts_epoch,
                                 ssim / ts_epoch), file=txtfile_ts)
                        print("epoch %d test|| d_loss: %.5f g_loss: %.5f adv_loss: %.5f s_loss: %.5f ssim: %.5f"
                              % (epoch_i, d_loss / ts_epoch, g_loss / ts_epoch, adv_loss / ts_epoch, s_loss / ts_epoch,
                                 ssim / ts_epoch))
                        # display
                        write_html(vis_dir, self.display_list)
                        print("============================")

                    # save model
                    if epoch_i % self.save_epoch == 0:
                        save_path = self.saver.save(sess, self.log_dir + '/gan_ckpt/gan', global_step=epoch_i)
                        print("Model saved in file: %s" % save_path)

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)
            sess.close()
        txtfile.close()
        txtfile_ts.close()

        return

    def build_inference(self, mode='test_gan', test_gallery=False, gallery_txt=None):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if self.crop_size is not None:
            self.image_size = self.crop_size
        else:
            self.image_size = self.input_image_size

        # placeholder
        self.train_mode = tf.placeholder(tf.bool, name='train_mode')

        # Input images
        # test batch
        self.photo_inp, self.sketch_inp, self.photo_identity, self.sketch_identity, self.photo_name, self.sketch_name, self.photo_num, self.sketch_num = \
            input_data.photo_sketch_batch_inputs(self.ts_inp_dir, self.ts_txt, self.ts_txt, self.num_identity, 1,
                                                 ['real_db_s'], self.ts_batch_size, img_size=self.input_image_size,
                                                 name='ts_inp', photo_dim=self.img_channels, sketch_dim=self.img_channels,
                                                 flip=False, crop_size=self.crop_size, padding_size=self.padding_size,
                                                 random_crop=False, train_mode='test_gan', concat_sketch_styles=True,
                                                 log_dir=self.log_dir+'/ts_inputs.txt')

#------------------
        #self.tr_sketch_inp = tf.roll (self.tr_sketch_inp, shift=2, axis=0)
#-----------------------
        # gallery batch
        if test_gallery:
            self.gallery_inp, self.gallery_identity, self.gallery_name, self.gallery_num = \
                input_data.photo_batch_inputs(self.ts_inp_dir + '/gallery', gallery_txt, 20000,
                                              self.ts_batch_size, img_size=self.input_image_size, name='gallery_inp',
                                              photo_dim=self.img_channels, flip=False, crop_size=self.crop_size,
                                              padding_size=self.padding_size, random_crop=False,
                                              train_mode='test_gallery',
                                              log_dir=self.log_dir + '/ts_gallery.txt')

        # build network
        self.build_network(mode=mode, test_gallery=test_gallery)

        return
