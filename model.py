
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, add, BatchNormalization, Activation, LeakyReLU

from subpixel_conv2d import SubpixelConv2D
from tensorflow.keras.models import Model

def get_G(input_shape):
	# w_init = tf.random_normal_initializer(stddev=0.02)
	g_init = tf.random_normal_initializer(1., 0.02)
	relu= Activation('relu')

	nin= Input(shape= input_shape)
	n= Conv2D(64, (3,3), padding='SAME', activation= 'relu',
						kernel_initializer='HeNormal')(nin)
	temp= n


	# B residual blocks
	for i in range(3):
		nn= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
		nn= BatchNormalization(gamma_initializer= g_init)(nn)
		nn= relu(nn)
		nn= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
		nn= BatchNormalization(gamma_initializer= g_init)(nn)

		nn= add([n, nn])
		n= nn

	n= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
	n= BatchNormalization(gamma_initializer= g_init)(n)
	n= add([n, temp])
	# B residual blacks end

	n= Conv2D(256, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
	n= SubpixelConv2D(upsampling_factor=2)(n)
	n= relu(n)

	n= Conv2D(256, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
	n= SubpixelConv2D(upsampling_factor=2)(n)
	n= relu(n)

	nn= Conv2D(3, (1,1), padding='SAME', kernel_initializer='HeNormal', activation= 'tanh')(n)


	G = Model(inputs=nin, outputs=nn, name="generator")
	return G



def get_D(input_shape):

	g_init= tf.random_normal_initializer(1., 0.02)
	ly_relu= LeakyReLU(alpha= 0.2)
	df_dim = 16

	nin = Input(input_shape)
	n = Conv2D(64, (4, 4), (2, 2), padding='SAME', kernel_initializer='HeNormal')(nin)
	n= ly_relu(n)

	for i in range(2, 6):
		n = Conv2D(df_dim*(2**i),(4, 4), (2, 2), padding='SAME', kernel_initializer='HeNormal')(n)
		n= ly_relu(n)
		n= BatchNormalization(gamma_initializer= g_init)(n)

	n= Conv2D(df_dim*16, (1, 1), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
	n= ly_relu(n)
	n= BatchNormalization(gamma_initializer= g_init)(n)

	n= Conv2D(df_dim*8, (1, 1), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
	n= BatchNormalization(gamma_initializer= g_init)(n)
	temp= n

	n= Conv2D(df_dim*4, (3, 3), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
	n= ly_relu(n)
	n= BatchNormalization(gamma_initializer= g_init)(n)

	n= Conv2D(df_dim*8, (3, 3), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
	n= BatchNormalization(gamma_initializer= g_init)(n)

	n= add([n, temp])

	n= Flatten()(n)
	no= Dense(units=1, kernel_initializer='HeNormal', activation= 'sigmoid')(n)
	D= Model(inputs=nin, outputs=no, name="discriminator")

	return D









