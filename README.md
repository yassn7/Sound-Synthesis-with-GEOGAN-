# Sound-Synthesis-with-GEOGAN-
Sound Synthesis with Geometric Generative Adversarial Network in tensorflow

sound synthesis via Generative Adversarial Network (GANs) is suggested. Both the generator and the discriminator consist of a deep neural network architecture combining fully connected and 1-d confolution layers. The generator learns how to synthesize audio samples from raw audio, initialized with Gaussian noise. The discriminator adds a support vector machine on top of the neural network,  uses mini batch discrimination  and hinge loss for trainig. The Itakura-Saito divergence is used as a cost function for training in an asynchrous weight update scheme 
the method is exemplified by synthesis of drum kick sounds from artificially created datasets and drum kick sounds from a given audio collection of electronic and acoustic kick drum sound
