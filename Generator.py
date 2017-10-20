from __future__ import division
import tensorflow as tf
import librosa
import numpy as np
from math import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pylab import *
from Utiles import *



def generator(z, batch_size, L, reuse_variables = None) : # shape z should be (batch_size,N,N)

	#Parameters : L, fwidth, out_channel_1_G, out_channel_2_G, out_channel_3_G, out_channel_4_G


	with tf.variable_scope(tf.get_variable_scope(), reuse = reuse_variables) as scope:


		#Fully connected layer 

		z = tf.reshape(z,[-1,L])

		g_w1 = tf.get_variable('g_w1', [L, L], initializer = tf.truncated_normal_initializer(stddev = 0.02)) 
		g_b1 = tf.get_variable('g_b1', [L], initializer = tf.constant_initializer(0))
		g1 = tf.add(tf.matmul(z,g_w1),g_b1) 
		g1 = tf.reshape(g1, [batch_size,-1, L])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='gbn1')
		g1 = tf.nn.relu(g1)


		
		#Perform 1d Convolution through the Sample 


		g_w2 = tf.get_variable('g_w2', [f_width, 1, out_channel_1_G], initializer = tf.truncated_normal_initializer(stddev = 0.02))
		g_b2 = tf.get_variable('g_b2', [out_channel_1_G], initializer = tf.constant_initializer(0))
		g2 = tf.reshape(g1, [-1, L, 1])
		g2 = tf.nn.conv1d(g2, g_w2, 1, 'SAME')
		g2 = g2 + g_b2
		g2 = tf.nn.relu(g2)
		g2 = tf.reshape(g2, [-1, 1, L, out_channel_1_G])
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='gbn2')
	


		g_w3 = tf.get_variable('g_w3', [f_width, out_channel_1_G, out_channel_2_G], initializer = tf.truncated_normal_initializer(stddev = 0.02))
		g_b3 = tf.get_variable('g_b3', [out_channel_2_G], initializer = tf.constant_initializer(0))
		g3 = tf.reshape(g2, [-1, L, out_channel_1_G])
		g3 = tf.nn.conv1d(g3, g_w3, 1, 'SAME')
		g3 = g3 + g_b3
		g3 = tf.nn.relu(g3)
		g3 = tf.reshape(g3, [-1, 1, L, out_channel_2_G]) #Reshape before Pooling
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='gbn3')
		


		g_w4 = tf.get_variable('g_w4', [f_width, out_channel_2_G, out_channel_3_G], initializer = tf.truncated_normal_initializer(stddev = 0.02))
		g_b4 = tf.get_variable('g_b4', [out_channel_3_G], initializer = tf.constant_initializer(0))
		g4 = tf.reshape(g3, [-1, L, out_channel_2_G])
		g4 = tf.nn.conv1d(g4, g_w4, 1, 'SAME')
		g4 = g4 + g_b4
		g4 = tf.nn.relu(g4)
		g4 = tf.reshape(g4, [-1, 1, L, out_channel_3_G]) #Reshape before Pooling
		g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='gbn4')
		


		g_w5 = tf.get_variable('g_w5', [f_width, out_channel_3_G, out_channel_4_G], initializer = tf.truncated_normal_initializer(stddev = 0.02)) #out_channel_4_G should be 1
		g_b5 = tf.get_variable('g_b5', [out_channel_4_G], initializer = tf.constant_initializer(0))
		g5 = tf.reshape(g4, [-1,L, out_channel_3_G])
		g5 = tf.nn.conv1d(g5, g_w5, 1, 'SAME')
		g5 = g5 + g_b5
		g5 = tf.nn.relu(g5)
		g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='gbn5')
	


		g_w6 = tf.get_variable('g_w6', [f_width, out_channel_4_G, out_channel_5_G], initializer = tf.truncated_normal_initializer(stddev = 0.02)) 
		g_b6 = tf.get_variable('g_b6', [out_channel_5_G], initializer = tf.constant_initializer(0))
		g6 = tf.reshape(g5, [-1, L, out_channel_4_G])
		g6 = tf.nn.conv1d(g6, g_w6, 1, 'SAME')
		g6 = g6 + g_b6
		g6 = tf.tanh(g6)
		g6 = tf.contrib.layers.batch_norm(g6, epsilon=1e-5, scope='gbn6')


		#Fully_connected layer + activation function tanh  to transform feature onto sample  


		g6 = tf.reshape(g6, [-1, L]) #reshape g5 before Matmul
		g_w7 = tf.get_variable('g_w7', [L, L], initializer = tf.truncated_normal_initializer(stddev = 0.02)) 
		g_b7 = tf.get_variable('g_b7', [L], initializer = tf.constant_initializer(0))
		g7 = tf.add(tf.matmul(g6,g_w7),g_b7) 
		g7 = tf.reshape(g7, [-1, L]) # Tensor of K*N samples 		
		g7 = tf.contrib.layers.batch_norm(g7, epsilon=1e-5, scope='gbn7')
		
		return g7






