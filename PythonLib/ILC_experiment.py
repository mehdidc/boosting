#! /usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy 
import numpy as np
import os
import multiprocessing
import argparse
import shutil
import math

from Lib.App.App import App
from Lib.App.AppEnv import AppEnv, AppEnvHandler
from Lib.App.AppParams import AppParams
from Lib.App.AppResult import AppResult, AppResultHandler
from Lib.App.AppTrainingMethod import AppTrainingMethod
from Lib.App.AppEnums import DataCategory, LossFunctionType, CorruptionType, TrainingMethod

from Lib.Utils.Normalize import parallelHistogramEqualization
from Lib.Utils.ProcessManager import ProcessManager
from Lib.Utils.StringCodeExec import createFunction

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample


from collections import Iterable


parser = None
args = None

def targetsMapKeyFromValue(targetsMap, value):
    if isinstance(value, str):
        if not value.isdigit():
            assert value in targetsMap.reverse_mapping.values()
            return createFunction("return targetsMap.%s" % value, "", additional_symbols=dict(targetsMap=targetsMap))()
    else:
        assert int(value) in targetsMap.reverse_mapping.keys()
        return int(value)

def NdArrayToArff(ndArray, filename, relationName, targetsMap=None, withClassName=True):
    assert isinstance(ndArray, numpy.ndarray) and len(ndArray.shape) == 2 and isinstance(filename, str) and os.path.splitext(filename)[1] == '.arff' and isinstance(relationName, str)

    file = open(filename, 'w')
    s = '@RELATION %s\n\n' % relationName
    for i in xrange(ndArray.shape[1] - 1):
        s += '@ATTRIBUTE value_%d NUMERIC\n' % i
    if targetsMap is not None:
        s += '@ATTRIBUTE class {'
        for key, val in targetsMap.reverse_mapping.items():
            s += (lambda : str(val) if withClassName else str(key))() + ','
        if s[-1] == ',' : s = s[:-1]
        s += '}\n'
    s += '\n'
    s += '@DATA\n'
    for i in xrange(ndArray.shape[0]):
        for j in xrange(ndArray.shape[1] - 1):
            s += str(ndArray[i,j]) + ','
        if targetsMap is not None:
            s += (lambda x : str(x) if withClassName else str(targetsMapKeyFromValue(targetsMap, x)))(ndArray[i, ndArray.shape[1] - 1]) + '\n'
        else:
            s = s[:-1]
    if s[-1] == '\n' : s = s[:-1]
    file.write(s)
    file.close()

def exportDataset(datasets, targetsMap, output_folder, withClassName=True):
    def export2arff(inputs, targets, targetsMap, filename, relationName, output_folder, withClassName=True):
        print inputs.shape, filename
        assert isinstance(inputs, numpy.ndarray)
        assert len(inputs.shape) == 2 
        assert isinstance(filename, str) and isinstance(relationName, str)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        while output_folder[-1] == '/':
            output_folder = output_folder[:-1]

        print '%s/%s' % (output_folder, filename)
        file = open('%s/%s' % (output_folder, filename), 'w')
        s = '@RELATION %s\n\n' % relationName
        for i in xrange(inputs.shape[1]):
            s += '@ATTRIBUTE value_%d NUMERIC\n' % i
        s += '@ATTRIBUTE class {'
        for key, val in targetsMap.reverse_mapping.items():
            s += (lambda x : str(x) if withClassName else str(key))(val) + ','
        if s[-1] == ',' : s = s[:-1]
        s += '}\n\n'
        s += '@DATA\n'
        for i in xrange(inputs.shape[0]):
            for j in xrange(inputs.shape[1]):
                s += str(inputs[i,j]) + ','
            s += (lambda x : str(x) if withClassName else str(targetsMapKeyFromValue(targetsMap, x)))(targets[i]) + '\n'
        if s[-1] == '\n' : s = s[:-1]
        file.write(s)
        file.close()
    p0 = multiprocessing.Process(target=export2arff,
        args=(numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]), datasets.Targets()[DataCategory.TRAIN], targetsMap, 'train.arff', 'TRAIN', output_folder, withClassName,))
    p1 = multiprocessing.Process(target=export2arff,
        args=(numpy.asarray(datasets.Inputs()[DataCategory.TEST]), datasets.Targets()[DataCategory.TEST], targetsMap, 'test.arff', 'TEST', output_folder, withClassName,))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

def MeanCancellation(X):
    assert isinstance(X, numpy.ndarray) and len(X.shape) == 2
    mean = numpy.mean(X, axis=0)
    new_X = numpy.asarray([x - mean for x in X])
    return new_X

def PCAWhitening(X, k=None, epsilon=1E-18):
    # the matrix X should be observations-by-components
    assert  isinstance(X, numpy.ndarray)\
        and len(X.shape) == 2

    X = X.T
#    print 'X.shape =', X.shape

    sigma = numpy.dot(X, X.T) / X.shape[1] - 1 # use of Bessel's Correction
#    print 'sigma.shape =', sigma.shape

    U,S,V = numpy.linalg.svd(sigma)
#    print 'U.shape =', U.shape
#    print 'S.shape =', S.shape
#    print 'V.shape =', V.shape
    if not k or k > U.shape[1]:
        k = U.shape[1]
    else:
        assert k > 0
#    print 'k =', k

    XRot = numpy.dot(U.T, X)  # Rotate version of the data
#    print 'XRot.shape =', XRot.shape
    XTilde = numpy.dot(U[:, 0:k].T, X) # reduced dimension representation of the data, where k is the number of eigenvectors to keep
#    print 'XTilde.shape =', XTilde.shape

    XPCAWhite = numpy.dot(numpy.diag(1. / numpy.sqrt(S + epsilon)), XRot)
#    print 'XPCAWhite.shape =', XPCAWhite.shape
    XZCAWhite = numpy.dot(U, XPCAWhite)
#    print 'XZCAWhite.shape =', XZCAWhite.shape

    return XPCAWhite.T, XZCAWhite.T

def BinarizingData(X):
    assert isinstance(X, numpy.ndarray) and len(X.shape) == 2

    def createHistogram(array):
        assert isinstance(array, numpy.ndarray) and len(array.shape) == 1
        hist = numpy.zeros((256,), dtype=int)
        bins = 0

        for val in array:
            if hist[val] == 0:
                bins += 1
            hist[val] += 1

        return hist, bins

    result = numpy.zeros_like(X)

    for i in xrange(X.shape[0]):
        observation = map(int, [x * 255 for x in X[i]])
        histogram, histogramBins = createHistogram(numpy.asarray(observation, dtype=int))
        maxE = -float("inf")
        threshold = 0
        for s in xrange(256):
            CS0 = [val for val in observation if val < s]
            NS0 = len(CS0)
            if NS0 != 0:
                part1 = 0.
                for j in xrange(0, s+1):
                    if histogram[j] != 0:
                        hjOverNS0 = float(float(histogram[j]) / float(NS0))
                        part1 += hjOverNS0 * math.log(hjOverNS0)
            else:
                part1 = float("inf")

            CS1 = [val for val in observation if val >= s]
            NS1 = len(CS1)
            if NS1 != 0:
                part2 = 0.
                for j in xrange(s+1, 256):
                    if histogram[j] != 0:
                        hjOverNS1 = float(float(histogram[j]) / float(NS1))
                        part2 += hjOverNS1 * math.log(hjOverNS1)
            else:
                part2 = float("inf")

            if math.isinf(part1) or math.isinf(part2):
                tmpE = -float("inf")
            else:
                tmpE = part1 - part2
            if tmpE > maxE:
                maxE = tmpE
                threshold = s
        assert not math.isinf(maxE)
        result[i] = numpy.asarray([(lambda x : 1 if x >= threshold else 0)(val) for val in observation], dtype=int)

    return result

class ILC_Env(AppEnv):
    def __init__(self):
        AppEnv.__init__(self,
            train=args.train,
            test=args.test,
            valid=args.validation)

class ILC_EnvHandler(AppEnvHandler):
    def __init__(self, app_env):
        assert isinstance(app_env, ILC_Env)
        AppEnvHandler.__init__(self, app_env)

class ILC_Params(AppParams):
    def __init__(self):
        AppParams.__init__(self,
            debug=args.debugMode)
        self.size = (18, 18, 30)
        self.offset = 0
        self.training_epochs = 15
        assert args.outputFolder is not None and isinstance(args.outputFolder, str)
        if args.outputFolder[-1] != '/':
            args.outputFolder += '/'
        self.output_folder += args.outputFolder

class ILC_Params_Benchmark(ILC_Params):
    def __init__(self):
        ILC_Params.__init__(self)
        ####################
        # Params Benchmark #
        ####################
        self.output_folder += 'Benchmark'

class ILC_Params_DAE(ILC_Params):
    def __init__(self):
        ILC_Params.__init__(self)
        self.training_epochs = 101
        ##################################
        # Params De-noising AutoEncoders #
        ##################################
        self.output_folder += 'DA'
        self.learning_rates = numpy.asarray([0.1])
        self.corruption_levels = numpy.asarray([0.3])
        self.hiddenLayer_size = numpy.asarray([1000])
        self.lossFunctionType = LossFunctionType.CROSS_ENTROPY
        self.corruptionType = CorruptionType.SALT_AND_PEPPER

class ILC_Params_SDAE(ILC_Params):
    def __init__(self):
        ILC_Params.__init__(self)
        self.training_epochs = 100
        ##########################################
        # Params Stacked De-noising AutoEncoders #
        ##########################################
        self.output_folder += 'SDAE'
        self.learning_rates = numpy.asarray([0.1, 0.1, 0.1])
        self.corruption_levels = numpy.asarray([0.3, 0.2, 0.1])
        self.hiddenLayer_size = numpy.asarray([1000, 500, 200])
        self.lossFunctionType = LossFunctionType.CROSS_ENTROPY
        self.corruptionType = CorruptionType.SALT_AND_PEPPER
    
    @staticmethod
    def from_dict(d):
        inst = ILC_Params_SDAE()
        inst.learning_rates = d["learning_rate"]
        inst.corruption_levels = numpy.asarray(d["corruption"])
        inst.hiddenLayer_size = numpy.asarray(sorted([int(n) for n in d["nb_neurons"]], reverse=True))
        inst.lossFunctionType = LossFunctionType.CROSS_ENTROPY
        inst.corruptionType = CorruptionType.SALT_AND_PEPPER
        inst.training_epochs = max(int(n) for n in d["nb_epochs"])
        return inst

class ILC_HP_Params(object):

    def __init__(self, templates=None, max_nb_layers=0, min_nb_layers=0, depthable=None):
        if templates is None:
            templates = {}
        self.templates = templates
        self.max_nb_layers = max_nb_layers
        self.min_nb_layers = min_nb_layers
        if depthable is None:
            dephtable = {}
        self.dephtable = depthable
    
    def get_hyper_args(self):
        args = []
        prec_vars = {}
        for depth in xrange(1, self.max_nb_layers + 1):
            vars = {}
            vars.update(prec_vars)
            prec_vars = vars
            vars["depth"] = depth
            for name, template in self.templates.items():
                name_cur_depth = name + "%d" % (depth,)
                vars[name_cur_depth] = template(name_cur_depth)
            args.append(vars)
        return args[self.min_nb_layers-1:]

    def get_values_from_dict(self, d):
        depth = d["depth"]

        values = {}
        for name in self.templates.keys():
            values[name] = [None] * depth

        for name, value in d.items():
            for template_name in self.templates.keys():
                if name.startswith(template_name):
                    layer = int(name[len(template_name):])
                    values[template_name][layer - 1] = value
        return values
    

class ILC_HP_Params_DBN(ILC_HP_Params):
    pass
class ILC_HP_Params_RBM(ILC_HP_Params):
    pass

class ILC_HP_Params_SDAE(ILC_HP_Params):
    def __init__(self):
        min_nb_layers = 3
        max_nb_layers = 4
        corruption = lambda name : hp.uniform(name, 0, 1)
        learning_rate = lambda name: hp.uniform(name, 0.5, 1)
        nb_neurons = lambda name: hp.quniform(name, 100, 800, 2)
        nb_epochs = lambda name: hp.quniform(name, 20, 50, 2)
        templates = {
                "corruption": corruption,
                "learning_rate": learning_rate,
                "nb_neurons": nb_neurons,
                "nb_epochs" : nb_epochs
        }
        ILC_HP_Params.__init__(self, templates, min_nb_layers=min_nb_layers, max_nb_layers=max_nb_layers)

class ILC_HP_Params_DAE(ILC_HP_Params):
    pass
 
class ILC_Params_RBM(ILC_Params):
    def __init__(self):
        ILC_Params.__init__(self)
        self.training_epochs = 3
        ###############
        # Params RBMs #
        ###############
        self.output_folder += 'RBM'
        self.learning_rate = numpy.asarray([0.1])
        self.nb_hidden = numpy.asarray([500])
        self.CD_k = numpy.asarray([15])

class ILC_Params_DBN(ILC_Params):
    def __init__(self):
        ILC_Params.__init__(self)
        self.training_epochs = 3
        ###############
        # Params DBNs #
        ###############
        self.output_folder += 'DBN'
        self.learning_rate = numpy.asarray([0.1, 0.1, 0.1])
        self.nb_hidden = numpy.asarray([1000, 500, 200])
        self.hiddenLayer_size = self.nb_hidden
        self.CD_k = numpy.asarray([15, 15, 15])

class ILC_Result(AppResult):
    def __init__(self):
        AppResult.__init__(self)

class ILC_ResultHandler(AppResultHandler):
    def __init__(self, app_result):
        assert isinstance(app_result, ILC_Result)
        AppResultHandler.__init__(self, app_result)

class ILC_TrainingMethod(AppTrainingMethod):
    def __init__(self):
        AppTrainingMethod.__init__(self,
            normalizeMethod=lambda x : parallelHistogramEqualization(x, min=0., max=1., processes=1))

class ILC_TrainingMethod_DAE(ILC_TrainingMethod):
    def __init__(self):
        ILC_TrainingMethod.__init__(self)
        self.method_type = TrainingMethod.DAE

class ILC_TrainingMethod_SDAE(ILC_TrainingMethod):
    def __init__(self):
        ILC_TrainingMethod.__init__(self)
        self.method_type = TrainingMethod.SDAE

class ILC_TrainingMethod_RBM(ILC_TrainingMethod):
    def __init__(self):
        ILC_TrainingMethod.__init__(self)
        self.method_type = TrainingMethod.RBM

class ILC_TrainingMethod_DBN(ILC_TrainingMethod):
    def __init__(self):
        ILC_TrainingMethod.__init__(self)
        self.normalizeMethod = lambda x : BinarizingData(ILC_TrainingMethod().normalizeMethod(x))
        self.method_type = TrainingMethod.DBN

class ILC_App(App):
    def __init__(self, app_params_class=ILC_Params_Benchmark, app_training_method_class=AppTrainingMethod, app_hp_params_class=ILC_HP_Params):
        App.__init__(self,
            app_env_class=ILC_Env,
            app_params_class=app_params_class,
            app_result_class=ILC_Result,
            app_training_method_class=app_training_method_class)
        self.hp_params = app_hp_params_class()

        global args
        if not isinstance(self.params, ILC_Params_Benchmark):
            args.withClassName = False

    def __MethodsPreProcess(self):
        assert self.datasets is not None
        trainData = numpy.asarray(self.datasets.Inputs()[DataCategory.TRAIN])

#        if not isinstance(self.params, ILC_Params_RBM):
#            self.params.DebugPrint('Mean Cancellation', self.params.printLvl + 1)
#            meanCancelTrainData = MeanCancellation(trainData)
#            trainData = meanCancelTrainData
#
#            self.params.DebugPrint('PCA Whitening', self.params.printLvl + 1)
#            whitenedTrainData, _ = PCAWhitening(trainData)
#            trainData = whitenedTrainData

        histogramMin, histogramMax = 0., 1.
        self.params.DebugPrint('Histogram Equalisation', self.params.printLvl + 1)
        self.params.DebugPrint('Min = %.3f' % histogramMin, self.params.printLvl + 2)
        self.params.DebugPrint('Max = %.3f' % histogramMax, self.params.printLvl + 2)
        equalizedTrainData = parallelHistogramEqualization(trainData, min=histogramMin, max=histogramMax)
        trainData = equalizedTrainData

        #if isinstance(self.params, ILC_Params_RBM) or isinstance(self.params, ILC_Params_DBN):
        #self.params.DebugPrint('Binarizing Data', self.params.printLvl + 1)
        binarizedTrainData = BinarizingData(trainData)
        trainData = binarizedTrainData

        self.datasets.SetInputs(trainData, DataCategory.TRAIN)

    def __MethodsProcess(self):
        assert self.datasets is not None
        
        res = self.training_method.Run(self.params, self.datasets, self.result.__class__)
        self.result = res
        return res
          
        def reduce_flatten(reduce_func, L):
            if isinstance(L, Iterable):
                return reduce_func(reduce_flatten(reduce_func, element) for element in L)
            else:
                return L
        def fn(hyperparams):
            print "Trying with %s..." % (str(hyperparams),)
            params_dict = self.hp_params.get_values_from_dict(hyperparams)
            params = self.params.__class__.from_dict(params_dict)
            ds = copy.deepcopy(self.datasets)
            res = self.training_method.Run(params, ds, self.result.__class__)
            sum_all_costs = reduce_flatten(sum, res.costs)
            return {"loss": sum_all_costs, "results": res, "status" : STATUS_OK}
        
        args = self.hp_params.get_hyper_args()
        
        min_nb_layers = self.hp_params.min_nb_layers
        max_nb_layers = self.hp_params.max_nb_layers

        space = scope.switch(
              hp.randint('d', max_nb_layers - min_nb_layers),
              *args
         )
        trials = Trials()
        best = fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        
        best_trial = min(trials.results, key=lambda res: res["loss"])
        self.result = best_trial["results"]
        print "Best params : %s, Loss : %.2f" % (str(best), best_trial["loss"],)
        #self.result = self.training_method.Run(self.params, self.datasets, self.result.__class__)
    
    def __MethodsPostProcess(self):
        assert self.result is not None
        if not os.path.isdir(self.params.output_folder):
            os.makedirs(self.params.output_folder)
        resultHdl = ILC_ResultHandler(self.result)
        resultHdl.Save(costs_filename='%s/costs.npy' % self.params.output_folder,
            filters_filename='%s/filters.npy' % self.params.output_folder)

    def __BenchmarkProcess(self):
        W, H, D = self.params.size
        

        def layers_difference(array):
            array = numpy.array(array).reshape((30, 18, 18))
            s = array.mean(axis=(1, 2))
            e = []
            for z in xrange(1, 30):
                e.append(s[z] - s[z - 1])
            return e
        
        def cov(array):
            array = numpy.array(array).reshape((30, 18, 18))
            a = array.mean(axis=0)
            
            x = numpy.arange(0, 18)
            x = x[:, np.newaxis]
            mu_x = (np.sum((x * a)) / np.sum(a))

            y = numpy.arange(0, 18)
            y = y[np.newaxis, :]

            mu_y = (np.sum((y * a)) / np.sum(a))
            cov_X_X = np.sum(((x - mu_x)**2) * a) / np.sum(a)
            cov_Y_Y = np.sum(((y - mu_y)**2) * a) / np.sum(a)
            cov_X_Y = np.sum( (x - mu_x) * (y - mu_y) * a ) / np.sum(a)
            return [cov_X_X, cov_Y_Y, cov_X_Y]

        def mean_energy(array, axis):
            array = numpy.array(array).reshape((30, 18, 18))
            size = (30, 18, 18)
            res = []
            for i in xrange(size[axis]):
                ind = [slice(0, s) for s in size]
                ind[axis] = slice(i, i + 1)
                res.append(numpy.mean(array[ind]))
            return res
        
        def benchmark_features(array):
            return layers_difference(array) + mean_energy(array, 0) + mean_energy(array, 1) + mean_energy(array, 2) + cov(array)

        # return a list of size W where each element x of this array is the sum over all
        # energies of points with X=x
        def toVectSum(array):
            # for 
            res = numpy.zeros((W,))
            for col in xrange(W):
                tmp = numpy.zeros((H,))
                for row in xrange(H):
                    for layer in xrange(D):
                        tmp[row] += array[col + W * (row + H * layer)]
                res[col] = numpy.sum(tmp)
            
            return res.tolist() + mean_energy(array, 0) + mean_energy(array, 1) + mean_energy(array, 2)

        # get a list of size W where each element x of this array is the mean
        # of the gradient of Y where each element of Y, y, is the mean of the
        # gradient of a vector of size H containing all the energies where X=x and Y=y
        def toVectGradient(array):
            res = numpy.zeros((W,))
            for col in xrange(W):
                tmp = numpy.zeros((H,))
                for row in xrange(H):
                    tmp2 = numpy.zeros((D,))
                    for layer in xrange(D):
                        tmp2[layer] = array[col + W * (row + H * layer)]
                    tmp[row] = numpy.mean(numpy.gradient(tmp2))
                res[col] = numpy.mean(numpy.gradient(tmp))
            return res.tolist()

        def toVectMean(array):
            res = numpy.zeros((W,))
            for col in xrange(W):
                tmp = numpy.zeros((H,))
                for row in xrange(H):
                    for layer in xrange(D):
                        tmp[row] += array[col + W * (row + H * layer)]
                    tmp[row] /= D
                res[col] = numpy.mean(tmp)
            return res.tolist()

        def hough_transform(img_bin, theta_res=1, rho_res=1):
            # Description : http://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/
            nR,nC = img_bin.shape
            theta = numpy.linspace(-90.0, 0.0, numpy.ceil(90.0/theta_res) + 1)
            theta = numpy.concatenate((theta, -theta[len(theta)-2::-1]))
            d = numpy.sqrt((nR - 1)**2 + (nC - 1)**2)
            q = numpy.ceil(d/rho_res)
            nrho = 2*q + 1
            rho = numpy.linspace(-q*rho_res, q*rho_res, nrho)
            H = numpy.zeros((len(rho), len(theta)))
            for rowIdx in range(nR):
                for colIdx in range(nC):
                    if img_bin[rowIdx, colIdx]:
                        for thIdx in range(len(theta)):
                            rhoVal = colIdx*numpy.cos(theta[thIdx]*numpy.pi/180.0) +\
                                rowIdx*numpy.sin(theta[thIdx]*numpy.pi/180)
                            rhoIdx = numpy.nonzero(numpy.abs(rho-rhoVal) == numpy.min(numpy.abs(rho-rhoVal)))[0]
                            H[rhoIdx[0], thIdx] += 1
            return rho, theta, H

        def houghTransform(array, threshold=0.):
            new_array = [(lambda x : x if x >= threshold else 0.)(y) for y in array]
            new_array = [int(x * 255) for x in new_array/numpy.linalg.norm(new_array)]
            assert all([isinstance(x, int) and 0 <= x <= 255 for x in new_array])
            pim = numpy.zeros((D,H))
            for layer in xrange(D):
                for row in xrange(H):
                    tmp = numpy.zeros((W,))
                    for col in xrange(W):
                        tmp[col] = new_array[col + W * (row + H * layer)]
                    pim[layer, row] = numpy.mean(tmp)
            # Calculate Hough transform.
            _,_,him = hough_transform(pim)
            return numpy.asarray(him).flatten().tolist()

        def ndArrayTreatment(ndArray, resultQueue, function):
            result = []
            for array in ndArray:
                tmp = function(array)
                result.append(tmp)
            resultQueue.put(numpy.asarray(result))

        def datasetTransform(datasets, resultsQueue, function):
            q0 = multiprocessing.Queue()
            p0 = multiprocessing.Process(target=ndArrayTreatment, args=(datasets.Inputs()[DataCategory.TRAIN], q0, function,))
            q1 = multiprocessing.Queue()
            p1 = multiprocessing.Process(target=ndArrayTreatment, args=(datasets.Inputs()[DataCategory.TEST], q1, function,))
            p0.start()
            trainResult = numpy.concatenate((q0.get(), numpy.asarray([datasets.Targets()[DataCategory.TRAIN]]).T), axis=1)
            p1.start()
            testResult = numpy.concatenate((q1.get(), numpy.asarray([datasets.Targets()[DataCategory.TEST]]).T), axis=1)
            p0.join()
            p1.join()
            resultsQueue.put(dict(train=trainResult, test=testResult))

        def datasetToVectSum(datasets, resultsQueue):
            datasetTransform(datasets, resultsQueue, benchmark_features)

        def datasetToVectGradient(datasets, resultsQueue):
            datasetTransform(datasets, resultsQueue, toVectGradient)

        def datasetToVectMean(datasets, resultsQueue):
            datasetTransform(datasets, resultsQueue, toVectMean)

        def datasetToHoughTransform(datasets, resultsQueue):
            datasetTransform(datasets, resultsQueue, houghTransform)

        def processingBenchmark(datasets, function, output_folder, relationName, targetsMap):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=function, args=(datasets, q,))
            p.start()
            p_results = q.get()
            p.join()
            pA = multiprocessing.Process(target=NdArrayToArff, args=(p_results['train'], '%s/train_benchmark_%s.arff' % (output_folder, relationName.replace('_', '')), 'TRAIN_%s' % relationName.upper(), targetsMap, args.withClassName,))
            pB = multiprocessing.Process(target=NdArrayToArff, args=(p_results['test'], '%s/test_benchmark_%s.arff' % (output_folder, relationName.replace('_', '')), 'TEST_%s' % relationName.upper(), targetsMap, args.withClassName,))

            pA.start()
            pB.start()
            pA.join()
            pB.join()

        self.params.DebugPrint('Computing Benchmark', self.params.printLvl + 1)

        if not os.path.isdir(self.params.output_folder):
            os.makedirs(self.params.output_folder)

        PManager = ProcessManager()

        p0 = PManager.add(multiprocessing.Process(target=processingBenchmark, args=(self.datasets, datasetToVectSum, self.params.output_folder, 'Vect_Sum', self.targetMap,)))
        p1 = PManager.add(multiprocessing.Process(target=processingBenchmark, args=(self.datasets, datasetToVectGradient, self.params.output_folder, 'Vect_Gradient', self.targetMap,)))
        p2 = PManager.add(multiprocessing.Process(target=processingBenchmark, args=(self.datasets, datasetToVectMean, self.params.output_folder, 'Vect_Mean', self.targetMap,)))
#        p3 = PManager.add(multiprocessing.Process(target=processingBenchmark, args=(self.datasets, datasetToHoughTransform, self.params.output_folder, 'Hough_Transform', self.targetMap,)))

        PManager.run()

        self.params.DebugPrint('Saving Benchmark', self.params.printLvl + 1)

    def _Load(self):
        assert  os.path.exists(args.train)\
            and (args.test is None or (args.test is not None and os.path.exists(args.test)))\
            and (args.validation is None or (args.validation is not None and os.path.exists(args.validation)))
        self.env.Train(new_train=args.train)
        if args.test is not None:
            self.env.Test(new_test=args.test)
        if args.validation is not None:
            self.env.Valid(new_valid=args.validation)

        envHdl = ILC_EnvHandler(self.env)
        self.datasets, self.targetMap = envHdl.Load(self.params, dtype=float, createTargetMap=True)

        while numpy.asarray(self.datasets.Inputs()[DataCategory.TRAIN]).shape[1] > numpy.prod(self.params.size):
            self.datasets.SetInputs([d[:-1] for d in self.datasets.Inputs()[DataCategory.TRAIN]], DataCategory.TRAIN)
        assert numpy.asarray(self.datasets.Inputs()[DataCategory.TRAIN]).shape[1] == numpy.prod(self.params.size)

    def _PreProcess(self):
        if not isinstance(self.params, ILC_Params_Benchmark):
            self.__MethodsPreProcess()

    def _Process(self):
        if isinstance(self.params, ILC_Params_Benchmark):
            self.__BenchmarkProcess()
        else:
            self.__MethodsProcess()

    def _PostProcess(self):
        if not isinstance(self.params, ILC_Params_Benchmark):
            self.__MethodsPostProcess()

def MainApp(parserArgs):
    global args
    args = parserArgs

    if args.runAll:
        args.benchmarkOnly = True
        args.methodsOnly = True

    def initMain():
        def addMethodToProcessManager(PManager, methods, methodName):
            assert isinstance(methodName, str) and isinstance(PManager, ProcessManager) and isinstance(methods, dict)

            getClassName = createFunction("""
            result = '%s_%s' % (className, method)
            return result
            """,
                "className, method")

            getClass = createFunction("""
            return eval(getClassName(className, method))
            """,
                "className, method",
                additional_symbols=dict(getClassName=getClassName, eval=eval, **globals()))

            methodProcess = createFunction("""
            return multiprocessing.Process(target=MyApp(app_params_class=MyAppParams, app_training_method_class=MyAppTrainingMethod, app_hp_params_class=MyAppHpParams).Run)
            """,
                additional_symbols=dict(MyApp=ILC_App,
                    MyAppParams=getClass('ILC_Params', methodName),
                    MyAppTrainingMethod=getClass('ILC_TrainingMethod', methodName),
                    MyAppHpParams=getClass('ILC_HP_Params', methodName),
                    **globals()
                )
            )()

            methodId = PManager.add(methodProcess)
            methods[methodName] = methodId, methodProcess

        PManager = ProcessManager()
        benchmark = PManager.add(multiprocessing.Process(target=ILC_App(app_params_class=ILC_Params_Benchmark).Run))

        methods = dict()
        if len(args.methodsList) == 0:
            args.methodsList = TrainingMethod.reverse_mapping.values()
        for methodName in args.methodsList:
            addMethodToProcessManager(PManager, methods, methodName)

        return PManager, benchmark, methods

    # PManager : process manager, will manage deep learning and benchmark processes
    # benchmark : 
    # methods : dict of name->(methodId, methodProcess) where methodProcess is a function
    #           doing deep learning for the "name" method
    PManager, benchmark, methods = initMain()
    if args.benchmarkOnly:
        PManager.run(benchmark)

    if args.methodsOnly:
        PManager.run([id for (id, p) in methods.values()])

def MainInitEnv(parserArgs):
    global args
    args = parserArgs
    assert  0. < args.ratioTrain < 1.\
        and os.path.exists(args.train)
    dir = os.path.dirname(args.train)
    setattr(args, 'test', None)
    setattr(args, 'validation', None)
    setattr(args, 'debugMode', True)

    env = ILC_Env()
    envHdl = ILC_EnvHandler(env)
    params = ILC_Params()

    datasets, targetMap = envHdl.Load(params, dtype=float, createTargetMap=True)

    while numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]).shape[1] > numpy.prod(params.size):
        datasets.SetInputs([d[:-1] for d in datasets.Inputs()[DataCategory.TRAIN]], DataCategory.TRAIN)
    assert numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]).shape[1] == numpy.prod(params.size)

    params.DebugPrint('Exporting Data')
    if os.path.exists('%s/train.arff' % dir):
        os.remove('%s/train.arff' % dir)
    if os.path.exists('%s/test.arff' % dir):
        os.remove('%s/test.arff' % dir)
    if args.ratioTrain == 1.:
        shutil.copy2(args.train, '%s/train.arff' % dir)
    elif args.ratioTrain == 0.:
        shutil.copy2(args.train, '%s/test.arff' % dir)
    else:
        params.DebugPrint('Ratio Train / All = %.3f %%' % (args.ratioTrain * 100), params.printLvl + 2)
        params.DebugPrint('Ratio Test / All = %.3f %%' % ((1. - args.ratioTrain) * 100), params.printLvl + 2)
        envHdl.TestFromTrain(datasets, ratioTrainTest=args.ratioTrain)
        exportDataset(datasets, targetMap, dir)

def MainProcessFilters(parserArgs):
    global args
    args = parserArgs
    setattr(args, 'debugMode', True)
    while args.methodDirectory[-1] == '/':
        args.methodDirectory = args.methodDirectory[:-1]
    while args.ArffDirectory[-1] == '/':
        args.ArffDirectory = args.ArffDirectory[:-1]
    assert os.path.isdir(args.methodDirectory) and os.path.isdir(args.ArffDirectory)
    print "method directory : %s "% (args.methodDirectory,)
    print args.ArffDirectory
    filters_filename = '%s/filters.npy' % args.methodDirectory
    costs_filename = '%s/costs.npy' % args.methodDirectory
    assert os.path.exists(filters_filename) and os.path.exists(costs_filename)
    result = ILC_Result()
    ILC_ResultHandler(result).Load(costs_filename, filters_filename)
    assert os.path.exists('%s/train.arff' % args.ArffDirectory)
    setattr(args, 'train', '%s/train.arff' % args.ArffDirectory)
    if os.path.exists('%s/test.arff' % args.ArffDirectory):
        setattr(args, 'test', '%s/test.arff' % args.ArffDirectory)
    else:
        setattr(args, 'test', None)
    if os.path.exists('%s/validation.arff' % args.ArffDirectory):
        setattr(args, 'validation', '%s/validation.arff' % args.ArffDirectory)
    else:
        setattr(args, 'validation', None)
    setattr(args, 'outputFolder', args.ArffDirectory)
    env = ILC_Env()
    envHdl = ILC_EnvHandler(env)
    params = ILC_Params()
    datasets, targetsMap = envHdl.Load(params, createTargetMap=True)
    def ComputeNewArray(datasets, category):
        for filter in result.filters:
            datasets.SetInputs(numpy.dot(numpy.asarray(datasets.Inputs()[category]), filter[0].T), category)
    if args.test is not None:
        ComputeNewArray(datasets, DataCategory.TEST)
    ComputeNewArray(datasets, DataCategory.TRAIN)

    if args.validation is not None:
        ComputeNewArray(datasets, DataCategory.VALIDATION)
    params.DebugPrint('Saving new Data')
    exportDataset(datasets, targetsMap, args.methodDirectory, withClassName=args.withClassName)
    for key, val in DataCategory.reverse_mapping.items():
        if os.path.exists('%s/%s.arff' % (args.methodDirectory, str(val).lower())):
            os.rename('%s/%s.arff' % (args.methodDirectory, str(val).lower()),
                '%s/%s_%s.arff' % (args.methodDirectory, str(val).lower(), os.path.basename(args.methodDirectory)))

def Main():
    global parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subParserA = subparsers.add_parser('run-app')
    subParserA.add_argument('--output-folder', dest='outputFolder', type=str, default='ILC/', required=False)
    subParserA.add_argument('--train', dest='train', type=str, default='./../Databases/ILC_DeepLearning_Experiment/Deep/train.arff', required=False)
    subParserA.add_argument('--test', dest='test', type=str, default='./../Databases/ILC_DeepLearning_Experiment/Deep/test.arff', required=False)
    subParserA.add_argument('--validation', dest='validation', type=str, default=None, required=False)
    subParserA.add_argument('-d', '--debug', dest='debugMode', action='store_true', default=False)
    groupA = subParserA.add_mutually_exclusive_group()
    groupA.add_argument('-a', '--run-all', dest='runAll', action='store_true', default=False)
    groupA.add_argument('-b', '--benchmark-only', dest='benchmarkOnly', action='store_true', default=False)
    groupA.add_argument('-m', '--methods-only', dest='methodsOnly', action='store_true', default=False)
    subParserA.add_argument('--methods', dest='methodsList', nargs='+', choices=TrainingMethod.reverse_mapping.values(), required=False, default=[])
    subParserA.add_argument('--with-class-name', dest='withClassName', action='store_true', default=False)
    subParserA.set_defaults(func=MainApp)

    subParserB = subparsers.add_parser('init-env')
    subParserB.add_argument('--arff-file', dest='train', type=str, default='./../Databases/ILC_DeepLearning_Experiment/Deep/PixelEnergy_2GeV_e6151_130207.arff', required=False)
    subParserB.add_argument('--ratio-train', dest='ratioTrain', type=float, default=0.8, required=False)
    subParserB.add_argument('--with-class-name', dest='withClassName', action='store_true', default=False)
    subParserB.set_defaults(func=MainInitEnv)

    subParserC = subparsers.add_parser('process-filters')
    subParserC.add_argument('methodDirectory', type=str)
    subParserC.add_argument('--arff-directory', dest='ArffDirectory', type=str, default='./../Databases/ILC_DeepLearning_Experiment/Deep/', required=False)
    subParserC.add_argument('--with-class-name', dest='withClassName', action='store_true', default=False)
    subParserC.set_defaults(func=MainProcessFilters)

    parserArgs = parser.parse_args()
    parserArgs.func(parserArgs)

if __name__ == '__main__':
    Main()
