#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
try:
    import cPickle as pickle
except:
    import pickle

from ..Utils.Print import Print

class AppParams(object):
    def __init__(self, debug=False):
        assert  isinstance(debug, bool)
        self.debug = debug
        self.printLvl = 0
        self.training_epochs = 15
        self.batch_size = 20
        self.output_folder = './../Outputs/'

    def IncrPrintLvl(self, val=1):
        assert isinstance(val, int)
        self.printLvl += val

    def DebugPrint(self, msg, printLvl=None, filename='log.txt'):
        print msg
        if self.debug:
            if printLvl is None:
                Print(msg, self.printLvl)
            else:
                assert isinstance(printLvl, int)
                Print(msg, printLvl)
        folder = self.output_folder
        while folder[-1] == '/':
            folder = folder[:-1]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if printLvl is None:
            Print(msg, self.printLvl, '%s/%s' % (folder, filename))
        else:
            assert isinstance(printLvl, int)
            Print(msg, printLvl, '%s/%s' % (folder, filename))

class AppParamsHandler(object):
    def __init__(self, app_params):
        assert isinstance(app_params, AppParams)
        self.params = app_params

    def Dump(self, filename):
        out_file = open(filename, 'w')
        pickle.dump(self.params, out_file)

    def Load(self, filename):
        assert os.path.exists(filename)