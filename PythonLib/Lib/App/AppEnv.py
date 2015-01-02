#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import os
import random

from AppParams import AppParams
from AppData import AppData
from AppEnums import DataCategory

from ..Utils.Enums import enum
from ..Utils.Print import Print
from ..Utils.StringCodeExec import createFunction

class AppEnv(object):
    def __init__(self, train, valid=None, test=None):
        assert  isinstance(train, str) \
                and (valid is None or isinstance(valid, str)) \
                and (test is None or isinstance(test, str))
                
        self.__train = train
        self.__valid = valid
        self.__test = test
        
    def Train(self, new_train=None):
        if new_train is not None:
            assert isinstance(new_train, str)
            self.__train = new_train
        return self.__train
            
    def Valid(self, new_valid=None):
        if new_valid is not None:
            assert isinstance(new_valid, str)
            self.__valid = new_valid
        return self.__valid
            
    def Test(self, new_test=None):
        if new_test is not None:
            assert isinstance(new_test, str)
            self.__test = new_test
        return self.__test
        
class AppEnvHandler(object):
    def __init__(self, app_env):
        assert isinstance(app_env, AppEnv)
        self.__env = app_env
        
    def Env(self, new_app_env=None):
        if new_app_env is not None:
            assert isinstance(new_app_env, str)
            self.__env = new_app_env
        return self.__env
        
    def Load(self, params, dtype=float, createTargetMap=False):
        assert isinstance(params, AppParams) and isinstance(dtype, type)
        params.DebugPrint('Loading Data')
        params.IncrPrintLvl()
        datasets, targetMap = self._Load(params, dtype, createTargetMap)
        params.IncrPrintLvl(-1)
        # targetMap is a "numbers" version of the targets
        return datasets, targetMap

    def _Load(self, params, dtype=float, createTargetMap=False):
        params.IncrPrintLvl()
        datasets = AppData(size=params.size, offset=params.offset)
        def loadData(filename, inputCategory):
            
            # load data from filename and put it in "datasets"
            # ----> set inputs AND target in datasets of the given input category
            if not os.path.exists("".join(os.path.splitext(filename)[:-1])+'.npy'):
                assert os.path.splitext(filename)[-1] == '.arff'
                inFile = open(filename)
                lines = inFile.readlines()
                tmp = [line.strip().split(',') for line in lines if line[0] != '@' and line[0] != '\n' and line != ' \n']
                post_tmp = [c for c in [[x.strip() for x in d] for d in tmp] if numpy.asarray(c[:-1], dtype=dtype).min() != numpy.asarray(c[:-1], dtype=dtype).max()]
                data = [[(lambda x : dtype(x))(x) for x in d[:-1]] for d in post_tmp]
                target = [d[-1] for d in post_tmp]
                numpy.save("".join(os.path.splitext(filename)[:-1])+'.npy', [data, target])
            else:
                tmp = numpy.load("".join(os.path.splitext(filename)[:-1])+'.npy')
                data, target = tmp[0].tolist(), tmp[1].tolist()
            datasets.SetInputs(data, inputCategory)
            datasets.SetTargets(target, inputCategory)

        # convert "targets" to numbers
        def createTargetsMap(params, datasets, inputSet=DataCategory.TRAIN):
            def mapTargets(datasets, inputSet, enumMapping):
                res = []
                for target in datasets.Targets()[inputSet]:
                    mapping = createFunction("return enumMapping.%s" % target, additional_symbols=dict(enumMapping=enumMapping))
                    res.append(mapping())
                datasets.SetTargets(res, inputSet)
            params.DebugPrint('Creating Targets Mapping')
            code = "enum("
            targets = list(set(datasets.Targets()[inputSet]))
            for target, i in zip(targets, xrange(len(targets))):
                code += "%s=%d," % (str(target), i)
            code = code[:-1] + ")"
            enumMapping = createFunction("return %s" % code, additional_symbols=dict(enum=enum))()
            mapTargets(datasets, inputSet, enumMapping)
            return enumMapping
        params.DebugPrint('Loading Train Set from "%s"' % self.Env().Train())
        loadData(self.Env().Train(), DataCategory.TRAIN)
        params.DebugPrint('%d example(s) of %d value(s)' % numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]).shape,\
            params.printLvl + 1)
        if createTargetMap:
            params.IncrPrintLvl()
            targetMap = createTargetsMap(params, datasets, DataCategory.TRAIN)
            params.DebugPrint(targetMap.reverse_mapping, params.printLvl + 1)
            params.IncrPrintLvl(-1)
        else:
            targetMap = None
        if self.Env().Valid() is not None:
            params.DebugPrint('Loading Validation Set from "%s"' % self.Env().Valid())
            loadData(self.Env().Valid(), DataCategory.VALIDATION)
            params.DebugPrint('%d example(s) of %d value(s)'
                              % numpy.asarray(datasets.Inputs()[DataCategory.VALIDATION]).shape,
                params.printLvl + 1)
        if self.Env().Test() is not None:
            params.DebugPrint('Loading Test Set from "%s"' % self.Env().Test())
            loadData(self.Env().Test(), DataCategory.TEST)
            params.DebugPrint('%d example(s) of %d value(s)'
                              % numpy.asarray(datasets.Inputs()[DataCategory.TEST]).shape,
                params.printLvl + 1)
        params.IncrPrintLvl(-1)
        return datasets, targetMap

    def TestFromTrain(self, datasets, ratioTrainTest=1.):
        # create test set from train set using ratioTrainTest (ratio=0.8 means 0.8train, 0.2test)
        # after creating test set, put it into datasets and "update" train set
        assert isinstance(ratioTrainTest, float) and 0 <= ratioTrainTest <= 1
        if ratioTrainTest != 1.:
            trainSet = []
            trainTargets = []
            testSet = []
            testTargets = []
            p = ratioTrainTest
            i = 0
            for x in datasets.Inputs()[DataCategory.TRAIN]:
                p = (ratioTrainTest * len(datasets.Inputs()[DataCategory.TRAIN]) - len(trainSet)) / (len(datasets.Inputs()[DataCategory.TRAIN]) - i)
                i += 1
                if random.random() < p:
                    trainSet.append(x)
                    trainTargets.append(datasets.Targets()[DataCategory.TRAIN][i - 1])
                else:
                    testSet.append(x)
                    testTargets.append(datasets.Targets()[DataCategory.TRAIN][i - 1])
            datasets.SetInputs(trainSet, DataCategory.TRAIN)
            datasets.SetTargets(trainTargets, DataCategory.TRAIN)
            datasets.SetInputs(testSet, DataCategory.TEST)
            datasets.SetTargets(testTargets, DataCategory.TEST)
