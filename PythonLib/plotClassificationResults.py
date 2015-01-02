#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pylab import *

benchmarkFolders = ['VectSum', 'VectMean', 'VectGradient']
methodFolders = ['DenoisingAutoEncoder', 'RestrictedBoltzmannMachine']
modelFolders = ['Tree', 'Product']
ErrorRateType = ['Train', 'Test']
templateDir = 'CrossValidation/Fold0'
MBDir = './../MultiBoost_Output/ILC'
benchmarksErrorRates = dict()
methodsErrorRates = dict()

def parseResultDta(filename):
    assert os.path.exists(filename)
    file = open(filename)
    errorRates = dict()
    for line in [l.split() for l in file.readlines()]:
         errorRates[int(line[0])] = (float(line[1]), float(line[2]))
    return errorRates

def getFolderErrorRates(folderName):
    result = dict()
    for model in modelFolders:
        modelDir = '%s/%s' % (templateDir, model)
        result[model] = parseResultDta('%s/%s/%s/results.dta' % (MBDir, folderName, modelDir))
    return result

def getMinMaxBenchmark(modelName):
    result = {'Min': benchmarkFolders[0], 'Max': benchmarkFolders[0]}
    for benchmark in benchmarkFolders:
        errorRates = sorted(benchmarksErrorRates[benchmark][modelName].values())
        tmp = {'Min':sorted(benchmarksErrorRates[result['Min']][modelName].values()),
               'Max':sorted(benchmarksErrorRates[result['Max']][modelName].values())}
        if errorRates[0][1] <= tmp['Min'][0][1]:
            result['Min'] = benchmark
        if errorRates[0][1] >= tmp['Max'][0][1]:
            result['Max'] = benchmark
    return result

def Plot(modelName, index):
    MinMaxBenchmark = getMinMaxBenchmark(modelName)

    YMinBenchmark = sorted(benchmarksErrorRates[MinMaxBenchmark['Min']][modelName].values(), reverse=True)
    XMinBenchmark = xrange(len(YMinBenchmark))
    plot(XMinBenchmark, [val[index] for val in YMinBenchmark], linewidth=2, linestyle="-", label="Benchmark")

    for method in methodFolders:
        YMethod = sorted(methodsErrorRates[method][modelName].values(), reverse=True)[:len(YMinBenchmark)]
        XMethod = xrange(len(YMethod))
        plot(XMethod, [val[index] for val in YMethod], linewidth=2, linestyle="-", label=method)

    legend(loc='upper right')

    xlabel("Iteractions")
    ylabel("Error Rate")

    title('%s Error Rate for %sLearner' % (ErrorRateType[index], modelName))
    savefig('%s/%s Error Rate for %sLearner.png' % (MBDir, ErrorRateType[index], modelName), dpi=72)
    show()

def main():
    global benchmarksErrorRates, methodsErrorRates
    for benchmark in benchmarkFolders:
        benchmarksErrorRates[benchmark] = getFolderErrorRates('benchmark_%s' % benchmark)
    for method in methodFolders:
        methodsErrorRates[method] = getFolderErrorRates(method)

    for index in xrange(len(ErrorRateType)):
        for model in modelFolders:
            Plot(model, index)

if __name__ == '__main__':
    main()