



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



def Train_GAN(dataset, n_epoch, n_pre_epoch, batch_size, wdecay, L,  NAME = None):


#Define the Placeholder 
	x = tf.placeholder(tf.float32, [None, L], name = 'training_sample_placeholder')
	z = tf.placeholder(tf.float32, [None, 1, L], name = 'input_noise_placeholder')


	Dreal,Freal = discriminator(x, batch_size ,L)
	Gz = generator(z, batch_size, L) 
	Dfake, Ffake = discriminator(Gz, batch_size,L, reuse_variables = True)




	tvars = tf.trainable_variables()

	weigts_d = tf.concat([tf.reshape(var,[1,-1]) for var in tvars if 'd_w' in var.name],1)
	Rd = wdecay

	weigts_g = tf.concat([tf.reshape(var,[1,-1]) for var in tvars if 'd_w' in var.name],1)
	Rg = wdecay

	d_loss_regularization = Rd*tf.norm(weigts_d,ord=2)
	d_loss_real = tf.losses.hinge_loss( logits = Dreal, labels = tf.ones_like(Dreal)) +  d_loss_regularization
	d_loss_fake = tf.losses.hinge_loss( logits = Dfake, labels = tf.zeros_like(Dfake)) +  d_loss_regularization
	eps = 0.0001
	g_loss_feature_matching = (tf.norm(Freal,ord = 2)/(tf.norm(Ffake,ord=2)+eps))  - tf.log(tf.norm(Ffake,ord = 2)/(tf.norm(Freal,ord =2)+ eps)) -1
	g_loss_L2 = Rg*tf.norm(weigts_g,ord=2)
	g_loss_regularization = g_loss_L2 + g_loss_feature_matching
	g_loss = tf.losses.hinge_loss(logits = Dfake, labels = tf.ones_like(Dfake))



	dvars = [var for var in tvars if 'd_' in var.name ]
	gvars = [var for var in tvars if 'g_' in var.name ]

	optimizer = tf.train.AdamOptimizer(1e-3)
	optimizer_g = tf.train.AdamOptimizer(1e-2)
	g_grad_var = optimizer_g.compute_gradients(g_loss_regularization,gvars)
	g_grad_var_clipped = [(tf.clip_by_value(grad,-1e5,1e5),var) for grad, var in g_grad_var]
	g_grad,g_var = [grad for grad,var in g_grad_var_clipped],[var for grad,var in g_grad_var_clipped]
	g_grad,_ = tf.clip_by_global_norm(g_grad,5.0)
	g_grad_var_clipped = [(grad,var) for grad,var in zip(g_grad,g_var)]
	g_trainer_regularization = optimizer.apply_gradients(g_grad_var_clipped)

	g_grad_var_2 = optimizer_g.compute_gradients(g_loss,gvars)
	g_grad_var_clipped_2 = [(tf.clip_by_value(grad,-1e5,1e5),var) for grad, var in g_grad_var_2]
	g_grad_2,g_var_2 = [grad for grad,var in g_grad_var_clipped_2],[var for grad,var in g_grad_var_clipped_2]
	g_grad_2,_ = tf.clip_by_global_norm(g_grad_2,5.0)
	g_grad_var_clipped_2 = [(grad,var) for grad,var in zip(g_grad_2,g_var_2)]
	g_trainer = optimizer.apply_gradients(g_grad_var_clipped_2)


	d_grad_var_real = optimizer.compute_gradients(d_loss_real,dvars)
	d_grad_var_clipped_real = [(tf.clip_by_value(grad,-1e5,1e5),var) for grad, var in d_grad_var_real]
	d_grad_real,d_var_real = [grad for grad,var in d_grad_var_clipped_real],[var for grad,var in d_grad_var_clipped_real]
	d_grad_real,_ = tf.clip_by_global_norm(d_grad_real,5.0)
	d_grad_var_clipped_real = [(grad,var) for grad,var in zip(d_grad_real,d_var_real)]
	d_trainer_real = optimizer.apply_gradients(d_grad_var_clipped_real)


	d_grad_var_fake = optimizer.compute_gradients(d_loss_fake,dvars)
	d_grad_var_clipped_fake = [(tf.clip_by_value(grad,-1e5,1e5),var) for grad, var in d_grad_var_fake]
	d_grad_fake,d_var_fake = [grad for grad,var in d_grad_var_clipped_fake],[var for grad,var in d_grad_var_clipped_fake]
	d_grad_fake,_ = tf.clip_by_global_norm(d_grad_fake,5.0)
	d_grad_var_clipped_fake = [(grad,var) for grad,var in zip(d_grad_fake,d_var_fake)]
	d_trainer_fake = optimizer.apply_gradients(d_grad_var_clipped_fake)

	tf.get_variable_scope().reuse_variables()

	
	saver = tf.train.Saver()

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess.run(tf.global_variables_initializer())


	predictionFake = np.zeros(batch_size)
	predictionReal = np.zeros(batch_size)
	for epoch in range(n_epoch):
		
		Saved_DlossReal = []
		Saved_Dreal = []
		Saved_DlossFake = []
		Saved_Dfake = []
		Saved_Gloss = []
		Saved_Gofz = []


		print('Epoch ',epoch,' over ',n_epoch)

			
		batch_number = 0
		for batch in dataset:

			batch_number+=1

			input_noise = np.random.normal(0,1,(batch_size,1,L))
			predictionReal = sess.run(Dreal,{x : batch})
			predictionFake = sess.run(Dfake,{z : input_noise })
			
			_,DlossReal,predictionReal = sess.run([d_trainer_real, d_loss_real, Dreal],{x : batch})
			Saved_DlossReal.append(DlossReal)
			Saved_Dreal.append(predictionReal)

			_,DlossFake,predictionFake = sess.run([d_trainer_fake, d_loss_fake, Dfake],{z : input_noise })
			Saved_DlossFake.append(DlossFake)
			Saved_Dfake.append(predictionFake)
						
			
			_,Gloss,GofZ, predictionFake = sess.run([g_trainer, g_loss, Gz, Dfake], {z : input_noise})
			_,GlossRegularization,GofZ = sess.run([g_trainer_regularization, g_loss_regularization, Gz], {x : batch, z : input_noise})
			GofZ = invMulaw(255,np.tanh(GofZ))
			Saved_Gloss.append(GlossRegularization)
			Saved_Gloss.append(Gloss)
			Saved_Gofz.append(GofZ)

		print('#############################################################################')
		print(NAME)
		print('#############################################################################')
		print('DLossFake = ', DlossFake)
		print('predictionFake  = ', np.mean(predictionFake))		
		print('DLossReal = ', DlossReal)
		print('predictionReal = ', np.mean(predictionReal))
		print('Gloss =', Gloss)
		print('GlossRegularization = ', GlossRegularization)
		print('max(Gofz) =', max(GofZ[0]))
		print('min(Gofz) = ', min(GofZ[0]))
		


		directory = '-'.join(('epoch',str(epoch)))
		np.save('_'.join((directory,'Gofz.npy',NAME)),Saved_Gofz)
		np.save('_'.join((directory,'Gloss.npy',NAME)),Saved_Gloss)
		np.save('_'.join((directory,'Dfake.npy',NAME)),Saved_Dfake)
		np.save('_'.join((directory,'Dreal.npy',NAME)),Saved_Dreal)
		np.save('_'.join((directory,'DlossReal.npy',NAME)),Saved_DlossReal)
		np.save('_'.join((directory,'DlossFake.npy',NAME)),Saved_DlossFake)
	
	save_path = saver.save(sess, NAME)
	
	sess.close()
	
	tf.reset_default_graph()
	




