#! /usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy
import time

class BoltzmannMachineParams(object):
    def __init__(self, numpy_rng, nb_visible, nb_hidden,
                 W=None, L=None, J=None,
                 bias_visible=None, bias_hidden=None):
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),
                                                        high=4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),\
                                                        size=(nb_visible, nb_hidden)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name="W", borrow=True)
            
        if not L:
            initial_L = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),
                                                        high=4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),\
                                                        size=(nb_visible, nb_hidden)),
                                      dtype=theano.config.floatX)
            numpy.fill_diagonal(initial_L, 0.)
            L = theano.shared(value=initial_L, name="L", borrow=True)
        
        if not J:
            initial_J = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),
                                                        high=4 * numpy.sqrt(6. / (nb_hidden + nb_visible)),\
                                                        size=(nb_visible, nb_hidden)),
                                      dtype=theano.config.floatX)
            numpy.fill_diagonal(initial_J, 0.)
            J = theano.shared(value=initial_J, name="J", borrow=True)
            
        self.W = W
        self.L = L
        self.J = J
            
        if not bias_visible:
            self.bias_visible = theano.shared(value=numpy.zeros(nb_visible,
                                                                dtype=theano.config.floatX),
                                         name="bias_visible",
                                         borrow=True)
                                        
        if not bias_hidden:
            self.bias_hidden = theano.shared(value=numpy.zeros(nb_hidden,
                                                               dtype=theano.config.floatX),
                                        name="bias_hidden",
                                        borrow=True)

        def Get(self):
            return [self.W, self.L, self.J, self.bias_hidden, self.bias_visible]

class BoltzmannMachine(object):
    def __init__(self, input, nb_visible, nb_hidden, params=None, numpy_rng=None, theano_rng=None):
        assert nb_visible > 0 and nb_hidden > 0  
        self.nb_visible = nb_visible
        self.nb_hidden = nb_hidden
        
        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(int(time.clock()))

        if theano_rng is None:
            theano_rng = theano.tensor.shared_randomstreams(numpy_rng.randint(2 ** 30))
            
        self._initParams(params, numpy_rng, nb_visible, nb_hidden)
        
        self.input = input
        if not input:
            self.input = theano.tensor.matrix('input')
            
        self.theano_rng = theano_rng
        
    def _initParams(self, params, numpy_rng, nb_visible, nb_hidden):
        if params is not None:
            assert isinstance(params, BoltzmannMachineParams)
            self.params = params
        else:
            self.params = BoltzmannMachineParams(numpy_rng, nb_visible, nb_hidden)
        