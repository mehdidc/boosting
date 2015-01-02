#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy

from Lib.App.AppResult import AppResult, AppResultHandler
from Lib.App.AppEnv import AppEnv, AppEnvHandler
from Lib.App.AppParams import AppParams
from Lib.App.AppEnums import DataCategory
from ILC_experiment import NdArrayToArff, ILC_Params

correctType = ['train', 'test', 'validation']
parser = None
args = None

def main():
    global parser, args
    parser = argparse.ArgumentParser()
    parser.add_argument('methodDirectory', type=str)
    parser.add_argument('arffFile', type=str)
    parser.add_argument('type', type=str)
    args = parser.parse_args()
    setattr(args, 'debugMode', True)
    while args.methodDirectory[-1] == '/':
        args.methodDirectory = args.methodDirectory[:-1]
    args.type = os.path.splitext(args.type)[0]
    filters_filename = '%s/filters.npy' % args.methodDirectory
    costs_filename = '%s/costs.npy' % args.methodDirectory
    arffFile = args.arffFile
    assert os.path.exists(filters_filename) and os.path.exists(costs_filename) and os.path.exists(arffFile) and args.type in correctType
    results = AppResult()
    AppResultHandler(results).Load(costs_filename, filters_filename)
    env = AppEnv(arffFile)
    dataset, targetsMap = AppEnvHandler(env).Load(ILC_Params(), createTargetMap=True)
    data, target = dataset.Inputs()[DataCategory.TRAIN], dataset.Targets()[DataCategory.TRAIN]
    print numpy.asarray(data).shape, results.filters.shape
#    ndArray = numpy.concatenate((numpy.asarray(data) * results.filters, numpy.asarray(target).T), axis=1)
#    NdArrayToArff(ndArray, '%s/%s.arff' % (args.methodDirectory, args.type), args.type.upper(), targetsMap)

if __name__ == '__main__':
    main()