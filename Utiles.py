

from __future__ import division
import tensorflow as tf
import librosa
import numpy as np
from math import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pylab import *





def mulaw(mu,X):
	
	y =np.array([np.sign(x)*(log(1+ int(mu*abs(x)))/log(1+mu)) for x in X])
	
	return y
	

def invMulaw(mu,Y):
	
	return np.array([np.sign(y)*((1+mu)**abs(y) -1)/mu for y in Y] )



def Read_dataset(batch_size):

	
	trainingSet = np.load('Kicks/Kicks.npy')
	trainingSet = [mulaw(255,y) for y in trainingSet]
	L = len(trainingSet[0])
	nsample = len(trainingSet)
	trainingSet = np.reshape(trainingSet,(int(nsample/batch_size),batch_size,L))


	return trainingSet,L

def sinwave(f,N,fs):
	s = [sin(2*pi*(f/fs)*i) for i in range(N)]
	return s

def Generate_Kicks(f,N,fs,tau):

	s = [exp(-i/(tau*fs))*sin(2*pi*(f/fs)*i) for i in range(N)]
	return s

def Generate_dataset(nsample, batch_size,L ,fs):

	# Generate train set : Kick samples
	print('Creating Dataset')
	trainingSet = [Generate_Kicks(f,L,fs,tau) for (tau,f) in  zip(np.random.uniform(0.001,0.002,nsample),np.random.uniform(500,1000,nsample))]
	trainingSet = np.reshape(trainingSet,(int(nsample/batch_size),batch_size,L))
	np.save('trainingSet.npy',np.array(trainingSet))
	print('Dataset Created')

	return trainingSet

def Generate_sinwaves(nsample, batch_size,L ,fs):

	# Generate train set : Kick samples
	print('Creating Dataset')
	trainingSet = [sinwave(f,L,fs) for f in  np.random.uniform(50,100,nsample)]
	trainingSet = np.reshape(trainingSet,(int(nsample/batch_size),batch_size,L))
	np.save('trainingSet.npy',np.array(trainingSet))
	print('Dataset Created')

	return trainingSet


f_width = 128
out_channel_1_G = 64
out_channel_2_G = 32
out_channel_3_G = 16
out_channel_4_G = 8
out_channel_5_G = 1

	




