#! /usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy

from ...Lib.Utils.Enums import *

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
        
        if input == None:
            self.X = theano.tensor.dmatrix(name="input")
        else:
            self.X = input
            
        self.params = [self.W, self.bias, self.bias_prime]
        
    def getHiddenValues(self, input):
        return theano.tensor.nnet.sigmoid(theano.tensor.dot(input, self.W) + self.bias)
        
    def getReconstructedInput(self, hidden):
        return theano.tensor.nnet.sigmoid(theano.tensor.dot(hidden, self.W_prime) + self.bias_prime)
        
    def getCostUpdates(self, learningRate, corruptionLevel, lossFunctionType=LossFunctionType.CROSS_ENTROPY):
        y = self.getHiddenValues(self.X)
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
            
        return (cost, updates)