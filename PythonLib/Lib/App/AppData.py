#! /usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import numpy

from AppEnums import DataCategory

class AppData(object):
    def __init__(self, size, offset=0,
                 trainSet=None, validationSet=None, testSet=None,
                 trainTargets=None, validationTargets=None, testTargets=None):
        if not trainSet: trainSet = []
        if not validationSet: validationSet = []
        if not testSet: testSet = []
        if not trainTargets: trainTargets = []
        if not validationTargets: validationTargets = []
        if not testTargets: testTargets = []

        assert isinstance(offset, int) and offset >= 0
        self.__offset = offset

        self.ResetSize(size)

        assert len(trainTargets) == len(trainSet)
        self.__trainSet = trainSet
        self.__trainTargets = trainTargets

        assert len(validationTargets) == len(validationSet)
        self.__validationSet = validationSet
        self.__validationTargets = validationTargets

        assert len(testTargets) == len(testSet)
        self.__testSet = testSet
        self.__testTargets = testTargets

    def Inputs(self):
        return self.__trainSet, self.__validationSet, self.__testSet
    def SetInputs(self, array, inputCategory):
        if inputCategory == DataCategory.TRAIN:
            self.__trainSet = array
        elif inputCategory == DataCategory.VALIDATION:
            self.__validationSet = array
        elif inputCategory == DataCategory.TEST:
            self.__testSet = array
    def Targets(self):
        return self.__trainTargets, self.__validationTargets, self.__testTargets
    def SetTargets(self, array, inputCategory):
        if inputCategory == DataCategory.TRAIN:
            self.__trainTargets = array
        elif inputCategory == DataCategory.VALIDATION:
            self.__validationTargets = array
        elif inputCategory == DataCategory.TEST:
            self.__testTargets = array
    def NormalizedSize(self):
        return self.__normalizeSize
    def Size(self):
        return self.__size
    def ResetSize(self, newSize, offset=None):
        assert isinstance(newSize, tuple)
        for val in newSize:
            assert isinstance(val, int) and val > 0
        self.__size = newSize
        if offset is not None:
            assert isinstance(offset, int) and offset >= 0
            self.__offset = offset
        self.__normalizeSize = tuple([x - (self.__offset * 2) for x in newSize])
    def Length(self):
        return len(self.__trainSet), len(self.__validationSet), len(self.__testSet)

class AppDataHandler(object):
    def __init__(self, app_data):
        assert isinstance(app_data, AppData)
        self.data = app_data

    def sharedDataset(self, inputCategory, borrow=True):
        data_x, data_y = self.data.Inputs()[inputCategory], self.data.Targets()[inputCategory]
        shared_x = theano.shared(numpy.asarray(data_x,
            dtype=theano.config.floatX),
            borrow=borrow
        )
        shared_y = theano.shared(numpy.asarray(data_y,
            dtype=theano.config.floatX),
            borrow=borrow
        )
        return shared_x, theano.tensor.cast(shared_y, 'int32')