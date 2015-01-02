#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy
import multiprocessing
from functools import partial
import sys
data_size = (18, 18, 30)
axisNames = ('I', 'J', 'K')
trainFilename = '../Databases/ILC_DeepLearning_Experiment/Deep/train.arff'
trainFilename = sys.argv[1]

trainOutputFilename = '../Databases/ILC_DeepLearning_Experiment/Deep/train_benchmark.arff'
testFilename = '../Databases/ILC_DeepLearning_Experiment/Deep/test.arff'
testFilename = sys.argv[2]
testOutputFilename = '../Databases/ILC_DeepLearning_Experiment/Deep/test_benchmark.arff'

sep = " "

def loadData(filename, saveToNpy=False):

    # si filename.npy (filename = sans le filename sans le .arff) n'existe pas alors...
    if not os.path.exists("".join(os.path.splitext(filename)[:-1])+'.npy'):
        assert os.path.splitext(filename)[-1] == '.arff'
        
        # lire le fichier arff dans "lines"
        inFile = open(filename)
        lines = inFile.readlines()
        # mettre dans tmp seulement les lignes de "lines" contenant les données
        # (cad sans @ au début puis ne prend pas les lignes vides)
        tmp = [line.strip().split(',') 
                for line in lines if line[0] != '@' and line[0] != '\n' and line != ' \n']
        # nettoyer les lignes de tmp avec un strip puis
        # filtrer les lignes de tmp qui ont toutes les features égales
        # et mettre tout ça dans post_tmp
        post_tmp = [c for c in [[x.strip() for x in d] for d in tmp] 
                    if (numpy.asarray(c[:-1], dtype=float).min() != 
                        numpy.asarray(c[:-1], dtype=float).max()) ]
        # data sera post_tmp sans la derniere colonne (qui est la classe)
        data = [[(lambda x : float(x))(x) for x in d[:-1]] for d in post_tmp]
        # target sera  la derniere colonne la classe
        target = [d[-1] for d in post_tmp]
        if saveToNpy:
            numpy.save("".join(os.path.splitext(filename)[:-1])+'.npy', [data, target])
    else:
        tmp = numpy.load("".join(os.path.splitext(filename)[:-1])+'.npy')
        data, target = tmp[0].tolist(), tmp[1].tolist()
    # data = les features, target = les classes (labels)
    return numpy.asarray(data), numpy.asarray(target)

def NormalizeInplace(array, imin=0, imax=1):
    dmin = array.min()
    dmax = array.max()
    array -= dmin
    array *= imax - imin
    array /= dmax - dmin
    array += imin

def NormalizeCopy(array, imin=0, imax=1):
    dmin = array.min()
    dmax = array.max()
    return imin + (imax - imin) * (array - dmin) / (dmax - dmin)

def Normalize2dArray(array):
    assert isinstance(array, numpy.ndarray) and len(array.shape) == 2
    res = []
    for row in array.astype(float):
        tmp = (row.astype(float) - row.astype(float).min()) / (row.astype(float).max() - row.astype(float).min())
        assert tmp.min() == 0. and tmp.max() == 1.
        res.append(tmp)
    return numpy.asarray(res)

def SimpleArrayHistogramEqualization(array, min, max):
    flattened_array = array.flatten()
    sorted_array = numpy.sort(flattened_array)
    cdf = sorted_array.cumsum() / sorted_array.max()
    current_min = min if min is not None else flattened_array.min()
    current_max = max if max is not None else flattened_array.max()
    NormalizeInplace(cdf, imin=current_min, imax=current_max)
    y = numpy.interp(flattened_array, sorted_array, cdf)
    new_array = y.reshape(array.shape)
    return new_array

def histogramEqualization(ndArray, min=None, max=None):
    assert isinstance(ndArray, numpy.ndarray) and len(ndArray.shape) == 2
    res = []
    for row in ndArray.astype(float):
        new_row = SimpleArrayHistogramEqualization(row, min, max)
        assert all([min <= val <= max for val in new_row])
        res.append(new_row)
    return numpy.asarray(res)

def parallelHistogramEqualization(ndArray, min=None, max=None, processes=1):
    assert processes >= 1
    if processes == 1:
        return histogramEqualization(ndArray, min=min, max=max)
    assert isinstance(ndArray, numpy.ndarray) and len(ndArray.shape) == 2 and isinstance(processes, int) and processes >= 1
    pool = multiprocessing.Pool(processes=processes)
    res = numpy.asarray(pool.map(partial(SimpleArrayHistogramEqualization, min=min, max=max), ndArray.astype(float)))
    assert all([min <= val <= max for val in res.flatten()])
    return res

def getVolume(X):
    res = numpy.zeros(data_size, dtype=float)
    for k in xrange(data_size[2]):
        for j in xrange(data_size[1]):
            for i in xrange(data_size[0]):
                res[i,j,k] = X[i + data_size[0] * (j + data_size[1] * k)]
    return res

def getArrayFromVolume(X):
    res = numpy.zeros(numpy.prod(data_size), dtype=float)
    for k in xrange(data_size[2]):
        for j in xrange(data_size[1]):
            for i in xrange(data_size[0]):
                res[i + data_size[0] * (j + data_size[1] * k)] = X[i,j,k]
    return res

def getFrontImage(X):
    assert X.shape == data_size
    return numpy.sum(X, axis=2)

def getLateralImage(X):
    assert X.shape == data_size
    return numpy.sum(X, axis=0)

def getRemainingAxis(axis):
    return sorted([a for a in xrange(len(data_size))
                   if a not in (lambda x : [x] if not isinstance(x, list) else x)(axis)], reverse=True)

def getWeights(X, axis):
    axis_left = getRemainingAxis(axis)
    assert len(axis_left) == 2
    res = X
    for a in axis_left:
        res = numpy.sum(res, a)
    assert res.shape == (data_size[axis],)
    return res

def getMean(X, axis, returnedWithWeights=False):
    assert X.shape == data_size
    weights = getWeights(X, axis)
    res = numpy.average(xrange(data_size[axis]), weights=weights)
    if returnedWithWeights:
        return res, weights
    else:
        return res

def getMeans(X):
    means = []
    for axis in xrange(len(data_size)):
        means.append(getMean(X, axis))
    return means

def computeVariance(X, axis):
    mean, weights = getMean(X, axis, returnedWithWeights=True)
    res = numpy.average(map(lambda x : numpy.square(x - mean), xrange(data_size[axis])), weights=weights)
    return res

def computeCovariance(X, axis1, axis2):
    if axis1 == axis2:
        return computeVariance(X, axis1)
    else:
        res = 0.
        axisOverWhichToSum = getRemainingAxis([axis1, axis2])
        assert len(axisOverWhichToSum) == 1 and axisOverWhichToSum[0] not in [axis1, axis2]
        new_X = numpy.sum(X, axis=axisOverWhichToSum[0])
        assert len(new_X.shape) == 2
        sumOverAxis1 = numpy.sum(new_X, axis=0)
        sumOverAxis2 = numpy.sum(new_X, axis=1)
        if data_size[axis1] != len(sumOverAxis2) or data_size[axis2] != len(sumOverAxis1):
            sumOverAxis1, sumOverAxis2 = sumOverAxis2, sumOverAxis1
        means = getMeans(X)
        for a1 in xrange(data_size[axis1]):
            for a2 in xrange(data_size[axis2]):
                res += (a1 - means[axis1]) * (a2 - means[axis2]) * sumOverAxis1[a2] * sumOverAxis2[a1]
        sumX = numpy.sum(X)
        assert sumX != 0.
        res /= sumX
        return res

def getCovarianceMatrix(X):
    res = numpy.zeros((len(data_size), len(data_size)), dtype=float)
    for axis1 in xrange(len(data_size)):
        for axis2 in xrange(axis1, len(data_size)):
            res[axis1, axis2] = computeCovariance(X, axis1, axis2)
            res[axis2, axis1] = res[axis1, axis2]
    return res

def computeCorrelation(covMatrix, axis1, axis2):
    assert covMatrix.shape == (len(data_size), len(data_size))
    varI = covMatrix[axis1, axis1]
    varJ = covMatrix[axis2, axis2]
    varITimesVarJ = varI * varJ
    return covMatrix[axis1, axis2] / numpy.sqrt(varITimesVarJ)

def getCorrelationMatrix(X):
    res = numpy.zeros((len(data_size), len(data_size)), dtype=float)
    covMatrix = getCovarianceMatrix(X)
    for axis1 in xrange(len(data_size)):
        for axis2 in xrange(len(data_size)):
            res[axis1, axis2] = computeCorrelation(covMatrix, axis1, axis2)
    return res


def exportToArff(output_filename, X, Y):
    def getHeader():
        # Relation Name Header
        relationNameDecl = "@RELATION" + sep + os.path.splitext(os.path.basename(output_filename))[0].upper()
        # Front Image Header
        frontImageDecl = []
        for i in xrange(data_size[0]):
            for j in xrange(data_size[1]):
                frontImageDecl.append('@ATTRIBUTE' + sep + 'FrontImage(' + ",".join((str(i + 1), str(j + 1))) + ')%sNUMERIC'%sep)
        # Lateral Image Header
        lateralImageDecl = []
        for j in xrange(data_size[1]):
            for k in xrange(data_size[2]):
                lateralImageDecl.append('@ATTRIBUTE' + sep + 'LateralImage(' + ",".join((str(j + 1), str(k + 1))) + ')%sNUMERIC'%sep)
        # Means Header
        meansDecl = []
        for axisName in axisNames:
            meansDecl.append('@ATTRIBUTE%sMean('%sep + axisName + ')%sNUMERIC'%sep)
        classesDecl = '@ATTRIBUTE%sclass%s{'%(sep,sep) + ",".join(sorted(set(Y))) + '}'
        # Result Header
        result =  relationNameDecl + '\n'              \
                + '\n'                                 \
                + "\n".join(frontImageDecl) + '\n'     \
                + "\n".join(lateralImageDecl) + '\n'   \
             #   + "\n".join(meansDecl) + '\n'          \
                + classesDecl
        return result
    def getData(X, Y):
        dataDecl = []
        for n in xrange(len(Y)):
            x,y = X[n], Y[n]
            tmpDataDecl = []
            frontImage = getFrontImage(x)
            for i in xrange(data_size[0]):
                for j in xrange(data_size[1]):
                    tmpDataDecl.append(str(frontImage[i,j]))
            assert len(tmpDataDecl) == data_size[0] * data_size[1]
            lateralImage = getLateralImage(x)
            for j in xrange(data_size[1]):
                for k in xrange(data_size[2]):
                    tmpDataDecl.append(str(lateralImage[j,k]))
            assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2])
            #means = getMeans(x)
            #for mean in means:
            #    tmpDataDecl.append(str(mean))
            #assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2]) + len(means)
            #covMatrix = getCovarianceMatrix(x)
            #for axis1 in xrange(len(axisNames)):
            #    for axis2 in xrange(axis1, len(axisNames)):
            #        tmpDataDecl.append(str(covMatrix[axis1, axis2]))
            #assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2]) + len(means) + numpy.sum(xrange(len(axisNames)))
            #corrMatrix = getCorrelationMatrix(x)
            #for axis1 in xrange(len(axisNames)):
            #    for axis2 in xrange(axis1, len(axisNames)):
            #        tmpDataDecl.append(str(corrMatrix[axis1, axis2]))
            #assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2]) + len(means) + 2*numpy.sum(xrange(len(axisNames)))
            #covMatrixEigenValues, _ = numpy.linalg.eig(covMatrix)
            #for n in xrange(len(axisNames)):
            #    tmpDataDecl.append(str(covMatrixEigenValues[n]))
            #assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2]) + len(means) + 2*numpy.sum(xrange(len(axisNames))) + len(axisNames)
            tmpDataDecl.append(y.strip())
            #assert len(tmpDataDecl) == data_size[1] * (data_size[0] + data_size[2]) + len(means) + 2*numpy.sum(xrange(len(axisNames))) + len(axisNames) + 1
            dataDecl.append(",".join(tmpDataDecl))
            print len(tmpDataDecl)
        result = "\n".join(dataDecl)
        return result
    header = getHeader()
    data = getData(X, Y)
    result =  header + '\n'  \
            + '\n'           \
            + '@DATA' + '\n' \
            + data
    output_file = open(output_filename, 'w')
    output_file.write(result)
    output_file.close()

def benchmark_run(output_filename, X, Y, withNormalization=True):
    new_X = map(lambda x : getVolume(x), (lambda x : parallelHistogramEqualization(x, min=0., max=1., processes=1) if withNormalization else x)(X))
    exportToArff(output_filename, new_X, Y)

def main():
    # Train Data, Labels
    train_X, train_Y = loadData(trainFilename)
    # Test Data, Labels
    test_X, test_Y = loadData(testFilename)

    # launch benchmark_run for train and test
    # ----> write benchmark arff files with LateralIMmage,FrontImage, Mean,etC.
    p0 = multiprocessing.Process(target=benchmark_run, args=(trainOutputFilename, train_X, train_Y,))
    p1 = multiprocessing.Process(target=benchmark_run, args=(testOutputFilename, test_X, test_Y,))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

if __name__ == '__main__':
    main()
