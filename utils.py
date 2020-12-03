import os, cv2
import numpy as np
from PIL import Image
import glob

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def load(path,shape):
	img= cv2.imread(path)
	img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img= cv2.resize(img, shape)
	return img


def process_LRI(path):
    if not os.path.exists('Data/LR'):
            os.makedirs('Data/LR')
            
    for i, file in enumerate(glob.glob(path + str('/*'))):
        img= Image.open(file)
        img= img.resize((96, 96))
        img.save('Data/LR/'+ str(i)+ '.png')


def process_HRI(path):
    if not os.path.exists('Data/HR'):
            os.makedirs('Data/HR')
            
    for i, file in enumerate(glob.glob(path + str('/*'))):
        img= Image.open(file)
        img= img.resize((96*4, 96*4))
        img.save('Data/HR/'+ str(i)+ '.png')


def get_data(path):
	X=[]
	Y=[]
	for folder in glob.glob(path+ str('/*')):
		for img_path in glob.glob(folder+ str('/*')):      
			if folder == os.path.join(path, 'HR'):
				X.append(load(img_path, (384, 384)))
			elif folder == os.path.join(path, 'LR'):
				Y.append(load(img_path, (96,96)))

	X= np.array(X)
	Y= np.array(Y)

	return X/255.0, Y/255.0



def get_vgg19():
	vgg= tf.keras.applications.VGG19( include_top=False, weights='imagenet', 
									input_tensor=None, input_shape=(384, 384, 3),
    								pooling=None, classes=1000, classifier_activation='softmax' )

	inp= Input(shape=(384, 384, 3))
	x= vgg.layers[0](inp)
	for ly in vgg.layers[1:17]:
	    x= ly(x)
	VGG19= Model(inp, x)

	return VGG19
	


