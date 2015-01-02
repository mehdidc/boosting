#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import os
import multiprocessing

class AppResult(object):
    def __init__(self):
        self.costs = []
        self.filters = []
    
class AppResultHandler(object):
    def __init__(self, app_result):
        assert isinstance(app_result, AppResult)
        self.result = app_result

    def __saveFunction(self, arr, filename):
        assert isinstance(filename, str) and os.path.splitext(filename)[1] == '.npy'
        numpy.save(filename, arr)
    def __saveCosts(self, filename) : self.__saveFunction(self.result.costs, filename)
    def __saveFilters(self, filename) : self.__saveFunction(self.result.filters, filename)

    def Save(self, costs_filename, filters_filename):
        p0 = multiprocessing.Process(target=self.__saveCosts, args=(costs_filename,))
        p1 = multiprocessing.Process(target=self.__saveFilters, args=(filters_filename,))
        p0.start()
        p1.start()
        p0.join()
        p1.join()

    def __loadFunction(self, filename, queue):
        assert isinstance(filename, str) and os.path.splitext(filename)[1] == '.npy' and os.path.exists(filename)
        queue.put(numpy.load(filename))

    def Load(self, costs_filename, filters_filename):
        q0 = multiprocessing.Queue()
        p0 = multiprocessing.Process(target=self.__loadFunction, args=(costs_filename, q0,))
        q1 = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=self.__loadFunction, args=(filters_filename, q1,))
        p0.start()
        self.result.costs = q0.get()
        p1.start()
        self.result.filters = q1.get()
        p0.join()
        p1.join()

    def Show(self):
        raise Exception("Not Implemented : Abstract class must be derivate")