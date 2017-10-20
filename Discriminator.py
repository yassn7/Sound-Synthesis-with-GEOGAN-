from __future__ import division
import functools
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import librosa
import numpy as np
from math import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
#from Utiles import *




def mb_discrimination(x,T,B,C,batch_size):

	M = tf.matmul(x,T)
	M = tf.reshape(M, [batch_size,B,C])
	ox = tf.zeros([1,1])
	for i in range (batch_size):
		Mi = tf.matmul(tf.ones([batch_size,1]),tf.reshape(M[i],[1,B*C]))
		Mi = tf.reshape(Mi, [batch_size,B,C])
		oxi = tf.exp(-tf.reduce_sum(tf.norm(Mi - M, ord = 1, axis = 2),0))
		ox = tf.concat([ox,[oxi]],1)
	ox = tf.slice(ox,[0,1],[1,batch_size*B])
	ox = tf.reshape(ox, [batch_size, -1])
	y =  tf.concat([x,ox],1)

	return y






def discriminator(sound, batch_size, L, reuse_variables = None):


	with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

		sound = tf.reshape(sound, [batch_size, L,1])
		

#		frame_length = int(44100 * 0.04)
#		overlap = 0.5
#		frame_step = int(frame_length - overlap*frame_length)
#		stft_sound = tf_contrib.signal.stft(sound,frame_length,frame_step,fft_length = frame_length ,window_fn =functools.partial(tf.contrib.signal.hann_window, periodic=True))
#		sound = tf.reshape(sound, [batch_size, L,1])
		#First Layer

		d_w1 = tf.get_variable('d_w', [64,1,64], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable('d_b', [64], initializer = tf.constant_initializer(0))
		d1 = tf.nn.conv1d(sound, d_w1, 1, 'SAME')
		d1 = d1 + d_b1
		d1 = tf.nn.relu(d1)
		d1 = tf.reshape(d1, [-1,batch_size,L,64])

		#Second Layer

		d_w2 = tf.get_variable('d_w2', [32,64,32], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [32], initializer = tf.constant_initializer(0))
		d1 = tf.reshape(d1, [-1,L,64])
		d2 = tf.nn.conv1d(d1, d_w2, 1, 'SAME')
		d2 = d2 + d_b2
		d2 = tf.nn.relu(d2)
		d2 = tf.reshape(d2, [batch_size,-1])

			#Mini Batch Discrimination
		B = 32
		C = 8
		T = tf.get_variable('T', [L*32,B*C], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d2 = mb_discrimination(d2,T,B,C, batch_size)

		
		#Third Layer

		d_w3 = tf.get_variable('d_w3', [16,32,16], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [16], initializer = tf.constant_initializer(0))
		d2 = tf.reshape(d2, [-1,L + 1,32])
		d3 = tf.nn.conv1d(d2, d_w3, 1, 'SAME')
		d3 = d3 + d_b3
		d3 = tf.nn.relu(d3)
		d3 = tf.reshape(d3, [-1,batch_size,L + 1,16])
		
			#Mini Batch Discrimination

		B = 16
		C = 8
		T16 = tf.get_variable('T16', [(L+1)*16,B*C], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d3 = tf.reshape(d3, [batch_size,-1])
		feature = d3 #Feature Matching
		d3_mbd =  mb_discrimination(d3,T16,B,C, batch_size)
		

		#Fourth Layer

		d_w4 = tf.get_variable('d_w4', [8,16,8], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [8], initializer = tf.constant_initializer(0))
		d3_mbd = tf.reshape(d3_mbd, [-1,L + 2,16])
		d4 = tf.nn.conv1d(d3_mbd, d_w4, 1, 'SAME')
		d4 = d4 + d_b4
		d4 = tf.nn.relu(d4)
		d4 = tf.reshape(d4, [-1,batch_size,L +2,8])

				#Mini Batch Discrimination
		B = 8
		C = 8
		T8 = tf.get_variable('T8', [(L+2)*8,B*C], initializer = tf.truncated_normal_initializer(stddev=0.02))
		d4 = tf.reshape(d4, [batch_size,-1])
		d4 = mb_discrimination(d4,T8,B,C, batch_size)


		#Fifth Layer

		d_w5 = tf.get_variable('d_w5', [8, 1], initializer = tf.truncated_normal_initializer(stddev=(0.02)))
		d_b5 = tf.get_variable('d_b5', [1], initializer = tf.constant_initializer(0))
		d5 = tf.reshape(d4, [-1, 8])
		d5 = tf.add(tf.matmul(d5,d_w5),d_b5)
		



		#Output Layers - fully Connected

#		d_w6 = tf.get_variable('d_w6', [4, 1], initializer = tf.truncated_normal_initializer(stddev=(0.02)))
#		d_b6 = tf.get_variable('d_b6', [1], initializer = tf.constant_initializer(0))
#		d6 = tf.add(tf.matmul(d5,d_w6),d_b6)
		d6 = tf.nn.tanh(d5)

		d_w7 = tf.get_variable('d_w7', [L + 3, 1], initializer = tf.truncated_normal_initializer(stddev=(0.02)))
		d_b7 = tf.get_variable('d_b7', [1], initializer = tf.constant_initializer(0))
		d6 = tf.reshape(d6, [batch_size,L + 3])
		d7 = tf.add(tf.matmul(d6,d_w7),d_b7)
		prediction = d7
		

		return prediction, feature
		


