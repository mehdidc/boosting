#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ..Utils.getNbChannels import *
from ..Utils.Data import *
from ..Utils.Enums import *
from ..Utils.PicturesShow import *

class DataHandler(object):
    
    def __init__(self):
        """ Initialize the parameters for the DataHandler Base Class
        """
        pass
    
    def sharedDataset(self, data, inputCategory, borrow=True):
        import theano, numpy
        data_x, data_y = data.Inputs()[inputCategory], data.Targets()[inputCategory]
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow
                                )
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow
                                )
        return shared_x, theano.tensor.cast(shared_y, 'int32')
        
    def mapTargets(self, data, inputCategory):
        raise Exception("DataHandler Base Class: MUST derivate the former base class and specialize the function mapTargets")
        
    def MeanCancel(self, data, inputCategory, shape=(0, 0), dims=2, nbChannels=1):
        inputs = numpy.asarray(data.Inputs()[inputCategory])
        assert (len(shape) == dims or len(shape) == dims + 1)
        assert len(inputs.shape) == 2 # inputs MUST be a squared matrix
        assert divmod(numpy.prod(shape), inputs.shape[1])[1] == 0
        (shape, nbChannels) = getNbChannels(shape, dims)
        
        Mean_shape = shape[1:]
        if nbChannels > 1:
            Mean_shape = shape + (nbChannels,)
        Mean = numpy.zeros(numpy.prod(Mean_shape), dtype=inputs.dtype)
                
        res = numpy.zeros_like(inputs)
        
        return res, Mean
        
#        ## Check the parameters
#        code = """
#assert (len(shape) == dims or len(shape) == dims + 1)
#assert len(inputs.shape) == 2 # inputs MUST be a squared matrix
#assert divmod(numpy.prod(shape), inputs.shape[1])[1] == 0
#(shape, nbChannels) = getNbChannels(shape, dims)
#        """
#        ## Compute Mean
#        code += """
#Mean_shape = shape
#if nbChannels > 1:
#    Mean_shape = shape + (nbChannels,)
#Mean = numpy.zeros(numpy.prod(Mean_shape), dtype=inputs.dtype)
#        """
#        ## Remove Mean from inputs
#        code += """
#res = numpy.zeros_like(inputs)
#        """
#        ## Return new inputs and mean
#        code += """
#print res.shape, Mean.shape, nbChannels
#return res, Mean
#        """
#        
#        from ..Utils.StringCodeExec import createFunction
#        from ..Utils.getNbChannels import getNbChannels
#        import numpy
#        internalMeanCancel = createFunction(code, "inputs, shape=(0, 0), dims=2, nbChannels=1",
#                                            additional_symbols = dict(numpy=numpy,
#                                                                      getNbChannels=getNbChannels))
#        
#        return internalMeanCancel(numpy.asarray(data.Inputs()[inputCategory]), shape, dims, nbChannels)

def test_MeanCancel():
    import numpy
    
    dims = 2
    shape = (28, 28)
    shape, nbChannels = getNbChannels(shape, dims)
    import time
    numpy.random.seed(int(time.clock()))
    inputs = numpy.random.random_integers(low=0, high=255, size=(7000, numpy.prod(shape)))
    targets = numpy.random.random_integers(low=0, high=9, size=(7000, 1))
    data = Data(normalizeSize=shape, trainSet=inputs, trainTargets=targets)
    new_inputs, Mean = DataHandler().MeanCancel(data, DataCategory.TRAIN, shape, dims, nbChannels)
    
    nb_lines = 100
    picturesShow = PicturesShow(npArray = inputs, imgShape=shape,
                                nbChannels=nbChannels, nbRows=nb_lines, tileSpace=(1, 1))
    picturesShow.Show()
    picturesShow = PicturesShow(npArray = new_inputs, imgShape=shape[:dims],
                                nbChannels=nbChannels, nbRows=nb_lines, tileSpace=(1, 1))
    picturesShow.Show()
    im_mean_size = shape
    if nbChannels > 1:
        im_mean_size = (shape + (nbChannels,))
    import Image
    im_mean = Image.new('L', im_mean_size)
    im_mean.putdata(numpy.reshape(Mean, numpy.prod(shape) * nbChannels).tolist())
    im_mean.show()