


import tensorflow as tf
import librosa
import numpy as np
from math import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pylab import *
from Utiles import *
from Generator import *
from Discriminator import *
from GAN import *


fs = 44100




batch_size = 5
nsample = 100

#dataset = Generate_dataset(nsample, batch_size,L ,fs)
dataset,L = Read_dataset(batch_size)

name ='_'.join(('Exp_L_',str(L)))
Train_GAN(dataset, 10000,0,batch_size,0.001, L,NAME = name)
