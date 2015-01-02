#! /usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy

from ...App.AppEnums import LossFunctionType, CorruptionType

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return theano.nnet.sigmoid(x)

class DenoisingAutoEncoder(object):        
    def __init__(self, numpy_randGenerator, theano_randGenerator=None, input=None,
                 nb_visible=28 * 28, nb_hidden=500, W=None, bias_hidden=None, bias_visible=None):
        self.nb_hidden = nb_hidden
        self.nb_visible = nb_visible
        
        if not theano_randGenerator:
            theano_randGenerator = theano.tensor.shared_randomstreams.RandomStreams(numpy_randGenerator.randint(2 ** 30))
        
        self.theano_randGenerator = theano_randGenerator
        self.numpy_randGenerator = numpy_randGenerator

        if not W:
            initial_W = numpy.asarray(numpy_randGenerator.uniform(low=-4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),
                                                                  high=4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),
                                                                  size=(nb_visible, nb_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name="W", borrow=True)
        
        if not bias_visible:
            bias_visible = theano.shared(value=numpy.zeros(nb_visible,
                                                          dtype=theano.config.floatX),
                                        name="bias_visible",
                                        borrow=True)
                                        
        if not bias_hidden:
            bias_hidden = theano.shared(value=numpy.zeros(nb_hidden,
                                                          dtype=theano.config.floatX),
                                        name="bias_hidden",
                                        borrow=True)
                                        
        self.W = W
        self.bias = bias_hidden
        self.bias_prime = bias_visible
        self.W_prime = self.W.T
        self.epoch = 1 
        self.nb_epochs = 100
        if input is None:
            self.X = theano.tensor.dmatrix(name="input")
        else:
            self.X = input
            
        self.params = [self.W, self.bias, self.bias_prime]
    def getCorruptedInput(self, input, corruptionLevel, corruptionType):
        def binomial_noise(theano_rng,inp,noise_lvl):
            """ This add binomial noise to inp. Only the salt part of pepper and salt.

            :type inp: Theano Variable
            :param inp: The input that we want to add noise
            :type noise_lvl: float
            :param noise_lvl: The % of noise. Between 0(no noise) and 1.
            """           
            assert isinstance(noise_lvl, float) and 0. <= noise_lvl <= 1.
            return theano_rng.binomial( size = inp.shape, n = 1, p = 1 - noise_lvl, dtype=theano.config.floatX) * inp

        def salt_and_pepper_noise(theano_rng,inp,noise_lvl):
            """ This add salt and pepper noise to inp

            :type inp: Theano Variable
            :param inp: The input that we want to add noise
            :type noise_lvl: tuple(float,float)
            :param noise_lvl: The % of noise for the salt and pepper. Between 0(no noise) and 1.
            """
            if not isinstance(noise_lvl, tuple):
                return binomial_noise(theano_rng, inp, noise_lvl)
            return  binomial_noise(theano_rng, theano.tensor.neq(inp, 0), noise_lvl[0]) \
                    + \
                    binomial_noise(theano_rng, theano.tensor.eq(inp, 0), 1 + noise_lvl[1])
#            return theano_rng.binomial( size = inp.shape, n = 1, p = 1 - noise_lvl[0], dtype=theano.config.floatX) * theano.tensor.neq(inp, 0) \
#                                + theano.tensor.eq(inp, 0) * theano_rng.binomial( size = inp.shape, n = 1, p = noise_lvl[1], dtype=theano.config.floatX)

        def gaussian_noise(theano_rng,inp,noise_lvl):
            """ This add gaussian NLP noise to inp

            :type inp: Theano Variable
            :param inp: The input that we want to add noise
            :type noise_lvl: float
            :param noise_lvl: The standard deviation of the gaussian.
            """
            assert isinstance(noise_lvl, float)
            return theano_rng.normal( size = inp.shape, std = noise_lvl, dtype=theano.config.floatX) + inp

        if corruptionType is CorruptionType.SALT_AND_PEPPER:
            return salt_and_pepper_noise(self.theano_randGenerator, input, corruptionLevel)
        elif corruptionType is CorruptionType.BINOMIAL_NOISE:
            return binomial_noise(self.theano_randGenerator, input, corruptionLevel)
        elif corruptionType is CorruptionType.GAUSSIAN_NOISE:
            return gaussian_noise(self.theano_randGenerator, input, corruptionLevel)
        else:
            raise Exception("CorruptionType not implemented !!!")
        
    def getHiddenValues(self, input):
        return theano.tensor.nnet.sigmoid(theano.tensor.dot(input, self.W) + self.bias)
        
    def getReconstructedInput(self, hidden):
        return theano.tensor.nnet.sigmoid(theano.tensor.dot(hidden, self.W_prime) + self.bias_prime)
    
    def getCostUpdates(self, learningRate, corruptionLevel, lossFunctionType=LossFunctionType.CROSS_ENTROPY, corruptionType=CorruptionType.SALT_AND_PEPPER):
        tilde_x = self.getCorruptedInput(self.X, corruptionLevel, corruptionType)
        y = self.getHiddenValues(tilde_x)
        z = self.getReconstructedInput(y)
        if lossFunctionType == LossFunctionType.CROSS_ENTROPY:
            L = - theano.tensor.sum(self.X * theano.tensor.log(z) + (1 - self.X) * theano.tensor.log(1 - z),
                                    axis=1)
        elif lossFunctionType == LossFunctionType.SQUARE:
            L = theano.tensor.square(self.X - z)
        else:
            raise Exception("Incorrect Loss Function Type !!!")
        cost = theano.tensor.mean(L)
        gparams = theano.tensor.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learningRate * gparam))
            
        return cost, updates
