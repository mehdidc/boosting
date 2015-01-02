#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import time
import theano

from .AppData import AppData, AppDataHandler
from .AppResult import AppResult
from .AppEnums import TrainingMethod, LossFunctionType, DataCategory

from ..Learner.AutoEncoders.DenoisingAutoEncoder import DenoisingAutoEncoder
from ..Learner.RestrictedBoltzmannMachine import RestrictedBoltzmannMachine

from ..Utils.StringCodeExec import createFunction
from ..Utils.Normalize import Normalize2dArray

class AppTrainingMethod(object):
    def __init__(self, method_type=None, nb_channels=1, normalizeMethod=Normalize2dArray):
        self.method_type = method_type
        self.nb_channels = nb_channels
        self.normalizeMethod = normalizeMethod
    
    def Run(self, app_params, datasets, app_result_class=AppResult):
        assert isinstance(datasets, AppData)
        app_params.DebugPrint('Training', app_params.printLvl + 1)
        app_params.IncrPrintLvl()
        app_params.DebugPrint('Parameters', app_params.printLvl + 1)
        for key, val in app_params.__dict__.items():
            if key not in ['debug', 'printLvl']:
                app_params.DebugPrint('{} : {}'.format(key, val), app_params.printLvl + 2)
        if self.method_type == TrainingMethod.DAE:
            app_params.DebugPrint('Method : Denoising AutoEncoder', app_params.printLvl + 1)
            app_params.IncrPrintLvl()
            result = self._TrainingDenoisingAutoEncoder(app_params, datasets, app_result_class)
        elif self.method_type == TrainingMethod.SDAE:
            app_params.DebugPrint('Method : Stacked Denoising AutoEncoders', app_params.printLvl + 1)
            app_params.IncrPrintLvl()
            result = self._TrainingStackedDenoisingAutoEncoders(app_params, datasets, app_result_class)
        elif self.method_type == TrainingMethod.RBM:
            app_params.DebugPrint('Method : Restricted Boltzmann Machine', app_params.printLvl + 1)
            app_params.IncrPrintLvl()
            result = self._TrainingRestrictedBoltzmannMachine(app_params, datasets, app_result_class)
        elif self.method_type == TrainingMethod.DBN:
            app_params.DebugPrint('Method : Deep Belief Network', app_params.printLvl + 1)
            app_params.IncrPrintLvl()
            result = self._TrainingDeepBeliefNetwork(app_params, datasets, app_result_class)
        else:
            raise Exception("The training method is not defined")
        app_params.IncrPrintLvl(-2)
        return result

    def __TrainingDeepArchitecture(self, app_params, datasets, app_result_class, model):
        code = """
        results = app_result_class()
        for layer in xrange(len(app_params.hiddenLayer_size)):
            if layer > 0:
                datasets.SetInputs(numpy.dot(numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]), numpy.asarray(results.filters)[layer - 1][0].T), DataCategory.TRAIN)
                datasets.ResetSize((app_params.hiddenLayer_size[layer - 1],))
            internalResults = self._Training%s(app_params, datasets, app_result_class, hiddenLayer=layer)
            results.filters.append(internalResults.filters)
            results.costs.append(internalResults.costs)
        return results
        """ % model
        return createFunction(code,
            args="app_params, datasets",
            additional_symbols=dict(self=self,
                DataCategory=DataCategory,
                numpy=numpy,
                AppResult=AppResult,
                app_result_class=app_result_class))(app_params, datasets)

    def _TrainingStackedDenoisingAutoEncoders(self, app_params, datasets, app_result_class):
        return self.__TrainingDeepArchitecture(app_params, datasets, app_result_class, 'DenoisingAutoEncoder')

    def _TrainingDeepBeliefNetwork(self, app_params, datasets, app_result_class):
        return self.__TrainingDeepArchitecture(app_params, datasets, app_result_class, 'RestrictedBoltzmannMachine')
    
    def _TrainingDenoisingAutoEncoder(self, app_params, datasets, app_result_class, hiddenLayer=0):
        code = """
        assert hiddenLayer >= 0

        app_params.IncrPrintLvl()

        if app_params.lossFunctionType == LossFunctionType.CROSS_ENTROPY and hiddenLayer > 0:
            datasets.SetInputs(self.normalizeMethod(numpy.asarray(datasets.Inputs()[DataCategory.TRAIN])), DataCategory.TRAIN)

        dataHdl = AppDataHandler(datasets)
        train_set_x, train_set_y = dataHdl.sharedDataset(DataCategory.TRAIN)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / app_params.batch_size

        # allocate symbolic variables for the data
        index = theano.tensor.lscalar() # index to a [mini]batch
        x = theano.tensor.matrix('X') # the data is presented as rasterized images

        rng = numpy.random.RandomState(int(time.clock()))
        theano_rng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))

        ######################
        # BUILDING THE MODEL #
        ######################
        app_params.DebugPrint("Building Model")
        assert self.nb_channels > 0
        da = DenoisingAutoEncoder(numpy_randGenerator=rng, theano_randGenerator=theano_rng,
                                  input=x, nb_visible=numpy.prod(datasets.Size()) * self.nb_channels,
                                  nb_hidden=app_params.hiddenLayer_size[hiddenLayer])
        da.nbepochs = app_params.training_epochs
        cost, updates = da.getCostUpdates(corruptionLevel=app_params.corruption_levels[hiddenLayer],
                                          learningRate=app_params.learning_rates[hiddenLayer],
                                          lossFunctionType=app_params.lossFunctionType,
                                          corruptionType=app_params.corruptionType)
        train_da = theano.function([index], cost,
                                   updates = updates,
                                   givens = {x: train_set_x[index * app_params.batch_size : (index + 1) * app_params.batch_size]}
                                   )
        ############
        # TRAINING #
        ############
        app_params.DebugPrint("Training Model")
        assert issubclass(app_result_class, AppResult)
        results = app_result_class()
        trainingCosts = []
        app_params.IncrPrintLvl()
        # go through training epochs
        start_time = time.clock()
        prec = None
        for epoch in xrange(1, app_params.training_epochs + 1): # go through training set
            c = []
            da.epoch = epoch
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
            #if prec is not None and prec < numpy.mean(c):
            #    print "Divergence...Stop"
            #    break
            costs = numpy.mean(c)
            lr = app_params.learning_rates[hiddenLayer]
            if prec is None or costs < prec:
                lr = min(0.9, lr + 0.1 * lr)
            else:
                lr = max(0.5, lr - 0.8 * lr)
            app_params.learning_rates[hiddenLayer] = lr
            prec = costs
            trainingCosts.append(numpy.mean(c))
            app_params.DebugPrint("Training epoch %d, cost " % (epoch) + str(numpy.mean(c)))
            app_params.DebugPrint("Learning rate : %.2f" % (lr,))
        end_time = time.clock()
        results.costs.append(numpy.asarray(trainingCosts))
        results.filters.append(da.W.get_value(borrow=True).T)
        training_time = (end_time - start_time)
        app_params.DebugPrint("Training took %.3f minutes" % (float(training_time) / 60.))

        app_params.IncrPrintLvl(-2)

        return results
        """
        return createFunction(code,
            args="app_params, datasets, hiddenLayer=0",
            additional_symbols=dict(self=self,
                LossFunctionType=LossFunctionType,
                DataCategory=DataCategory,
                numpy=numpy,
                time=time,
                theano=theano,
                AppDataHandler=AppDataHandler,
                AppResult=AppResult,
                app_result_class=app_result_class,
                DenoisingAutoEncoder=DenoisingAutoEncoder))(app_params, datasets, hiddenLayer)
        
    def _TrainingRestrictedBoltzmannMachine(self, app_params, datasets, app_result_class, hiddenLayer=0):
        code = """
        assert hiddenLayer >= 0

        app_params.IncrPrintLvl()

        if hiddenLayer > 0:
            datasets.SetInputs(self.normalizeMethod(numpy.asarray(datasets.Inputs()[DataCategory.TRAIN])), DataCategory.TRAIN)

        dataHdl = AppDataHandler(datasets)
        train_set_x, train_set_y = dataHdl.sharedDataset(DataCategory.TRAIN)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / app_params.batch_size

        # allocate symbolic variables for the data
        index = theano.tensor.lscalar() # index to a [mini]batch
        x = theano.tensor.matrix('X') # the data is presented as rasterized images

        rng = numpy.random.RandomState(int(time.clock()))
        theano_rng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))

        # initialize storage for the persistent chain (state = hidden layer of chain)
        persistent_chain = theano.shared(numpy.zeros((app_params.batch_size, app_params.nb_hidden[hiddenLayer]),
                                                     dtype=theano.config.floatX),
                                         borrow=True)

        ######################
        # BUILDING THE MODEL #
        ######################
        app_params.DebugPrint("Building Model")
        rbm = RestrictedBoltzmannMachine(input=x, nb_visible=numpy.prod(datasets.Size()) * self.nb_channels,
                                         nb_hidden=app_params.nb_hidden[hiddenLayer], numpy_rng=rng, theano_rng=theano_rng)

        # get the cost and the gradient corresponding to one step
        cost, updates = rbm.get_cost_updates(lr=app_params.learning_rate[hiddenLayer],
                                             persistent=persistent_chain, k=app_params.CD_k[hiddenLayer])

        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function([index], cost,
               updates=updates,
               givens={x: train_set_x[index * app_params.batch_size:
                                      (index + 1) * app_params.batch_size]},
               name='train_rbm')

        ############
        # TRAINING #
        ############
        app_params.DebugPrint("Training Model")
        assert issubclass(app_result_class, AppResult)
        results = app_result_class()
        trainingCosts = []
        app_params.IncrPrintLvl()
        # go through training epochs
        start_time = time.clock()
        for epoch in xrange(app_params.training_epochs): # go through training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_rbm(batch_index))
            trainingCosts.append(numpy.mean(c))
            app_params.DebugPrint("Training epoch %d, cost " % (epoch + 1) + str(numpy.mean(c)))
        end_time = time.clock()
        results.costs.append(numpy.asarray(trainingCosts))
        results.filters.append(rbm.params.W.get_value(borrow=True).T)
        training_time = end_time - start_time
        app_params.DebugPrint('Training took %f minutes' % (training_time / 60.))

        app_params.IncrPrintLvl(-2)

        return results
        """
        return createFunction(code,
            args="app_params, datasets, hiddenLayer=0",
            additional_symbols=dict(self=self,
                DataCategory=DataCategory,
                numpy=numpy,
                time=time,
                theano=theano,
                AppDataHandler=AppDataHandler,
                AppResult=AppResult,
                app_result_class=app_result_class,
                RestrictedBoltzmannMachine=RestrictedBoltzmannMachine))(app_params, datasets, hiddenLayer)
