import os
import time
import random
import numpy as np
from PIL import Image


import tensorflow as tf

from model import get_G, get_D
from utils import *

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = 5  
lr_v = 1e-3
tot_sample= 100  # Totall traning images
## adversarial learning (SRGAN)

n_epoch = 500
n_epoch_init = n_epoch//5

# create folders to save result images and trained models
save_dir = "samples"
checkpoint_dir = "checkpoint"
save_ind= 0

if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)


def train():
	# load Data
	HR_train, LR_train= get_data('Data')
	print('Data Loaded')
	# load modols
	G = get_G((96, 96, 3))
	D = get_D((384, 384, 3))
	vgg= get_vgg19()
	print('Models Loaded')
	# Optimizers
	g_optimizer_init = tf.optimizers.Adam(lr_v)
	g_optimizer = tf.optimizers.Adam(lr_v)
	d_optimizer = tf.optimizers.Adam(lr_v)


	print('Processing for initial G. learning')
	## initialize learning (G)
	for epoch in range(n_epoch_init):
		i,j= ((epoch)*batch_size)%tot_sample, (((epoch+1))*batch_size)%tot_sample
		if j== 0:
			j = -1
		X, Y= LR_train[i: j], HR_train[i: j]
		with tf.GradientTape() as tape:
			ypred= G(X)
			mse_loss= tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(Y, ypred), axis=-1))
			grad = tape.gradient(mse_loss, G.trainable_weights)
			g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
	        
		print("Epoch: [{}/{}] step: mse: {:.3f} ".format(
	            epoch, n_epoch_init , mse_loss))
		if epoch%10 ==0:
		    img= G.predict(LR_train[np.newaxis, save_ind])[0]
		    img= (img-img.mean())/img.std()
		    img= Image.fromarray(np.uint8(img*255))
		    img.save(os.path.join(save_dir, 'train_g_init{}.png'.format(epoch)))


	print('Processing for initial D. learning')
	for epoch in range(n_epoch):
		i,j= ((epoch)*batch_size)%tot_sample, (((epoch+1))*batch_size)%tot_sample
		if j== 0:
			j= -1
		X, Y= LR_train[i: j], HR_train[i: j]
		with tf.GradientTape(persistent=True) as tape:
			fake_img= G(X)
			fake_logits= D(fake_img)
			real_logits= D(Y)
			fake_feature= vgg(fake_img)
			real_feature= vgg(Y)

			#D. loss
			d_loss1= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits , tf.zeros_like(fake_logits)))
			d_loss2= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logits,tf.ones_like(real_logits)))
			d_loss= d_loss1 + d_loss2

			#G. loss
			g_gan_loss= 1e-3 *tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logits , tf.ones_like(fake_logits)))
			mse_loss= (1/(16*96*96)) *tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(fake_img, Y), axis=-1))
			vgg_loss = (1/(96*96))* tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(fake_feature, real_feature), axis=-1))
			g_loss = mse_loss + vgg_loss + g_gan_loss

			#Optimize
			grad = tape.gradient(g_loss, G.trainable_weights)
			g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
			grad = tape.gradient(d_loss, D.trainable_weights)
			d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

		print("Epoch: [{}/{}] step: D.loss: {:.3f}: G.loss: {:.3f}".format(
	            epoch, n_epoch , d_loss, g_loss))


		if epoch%50 ==0:
			img= G.predict(LR_train[np.newaxis, save_ind])[0]
			# if not sigmoid
			#img= (img-img.mean())/img.std()
			img= Image.fromarray(np.uint8(img*255))
			img.save(os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))

			G.save(os.path.join(checkpoint_dir, 'train_G_{}.h5'.format(epoch)))



if __name__== '__main__':
	train()