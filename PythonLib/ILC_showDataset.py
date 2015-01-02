#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import argparse
import random
import os

from Lib.App.AppEnv import AppEnv, AppEnvHandler
from Lib.App.AppParams import AppParams
from Lib.App.AppEnums import DataCategory
from Lib.Utils.Normalize import histogramEqualization

from mayavi import mlab

defaultSize = (18, 18, 30)
parser = None
args = None
listIndex = None

def arrayTo3dArray(array):
    assert isinstance(array, numpy.ndarray) and len(array.shape) == 1 and array.shape[0] == numpy.prod(args.size)
    W, H, D = args.size
    result = numpy.zeros((D, H, W), dtype=float)
    for layer in xrange(D):
        for row in xrange(H):
            for col in xrange(W):
                result[D-1-layer, row, col] = array[col + W * (row + H * layer)]
    return result

def getRandomArray(ndArray, excludeIndexList=[]):
    assert isinstance(ndArray, numpy.ndarray) and len(ndArray.shape) == 2
    index = random.randint(0, ndArray.shape[0] - 1)
    while index in excludeIndexList:
        index = random.randint(0, ndArray.shape[0] - 1)
    excludeIndexList.append(index)
    return index, ndArray[index]

def loadFromArff(filename):
    env = AppEnv(filename)
    envHdl = AppEnvHandler(env)
    params = AppParams()
    params.__setattr__('size', args.size)
    params.__setattr__('offset', 0)
    datasets, targetMap = envHdl.Load(params, dtype=float, createTargetMap=True)
    while numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]).shape[1] > numpy.prod(params.size):
        datasets.SetInputs([d[:-1] for d in datasets.Inputs()[DataCategory.TRAIN]], DataCategory.TRAIN)
    assert numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]).shape[1] == numpy.prod(params.size)
    return numpy.asarray(datasets.Inputs()[DataCategory.TRAIN]), numpy.asarray(datasets.Targets()[DataCategory.TRAIN]), targetMap

def saveData(filename, array):
    assert isinstance(array, numpy.ndarray) and len(array.shape) == 1 and array.shape[0] == numpy.prod(args.size)
    file = open(filename, 'w')
    s = ""
    for val in array:
        s += str(val) + ' '
    while s[-1] == ' ':
        s = s[:-1]
    file.write(s)
    file.close()

def ImproveDataDisplay():
    # Changing the ctf:
    from tvtk.util.ctf import ColorTransferFunction
    ctf = ColorTransferFunction()
    ctf.range = [0, 1]

    # Add points to CTF
    # Note : ctf.add_rgb_point(value, r, g, b) r, g, b are float numbers in [0,1]
    ctf.add_rgb_point(0, 0, 1, 0)
    ctf.add_rgb_point(0.7, 1, 0, 0)
    ctf.add_rgb_point(1, 1, 0, 0)

    # Changing the otf:
    from tvtk.util.ctf import PiecewiseFunction
    otf = PiecewiseFunction()

    # Add points to OTF
    # Note : otf.add_point(value, opacity) -> opacity is a float number in [0,1]
    otf.add_point(0, 0.015)
    otf.add_point(0.2, 1)
    otf.add_point(1, 1)

    return ctf, otf

def ImproveFilterDisplay():
    # Changing the ctf:
    from tvtk.util.ctf import ColorTransferFunction
    ctf = ColorTransferFunction()
    ctf.range = [-1, 1]

    # Add points to CTF
    # Note : ctf.add_rgb_point(value, r, g, b) r, g, b are float numbers in [0,1]
    ctf.add_rgb_point(-1, 0, 0, 1)
    ctf.add_rgb_point(-0.7, 0, 0, 1)
    ctf.add_rgb_point(0, 0, 1, 0)
    ctf.add_rgb_point(0.7, 1, 0, 0)
    ctf.add_rgb_point(1, 1, 0, 0)

    # Changing the otf:
    from tvtk.util.ctf import PiecewiseFunction
    otf = PiecewiseFunction()

    # Add points to OTF
    # Note : otf.add_point(value, opacity) -> opacity is a float number in [0,1]
    otf.add_point(-1, 0.005)
    otf.add_point(0, 0.01)
    otf.add_point(0.6, 0.1)
    otf.add_point(1, 1)

    return ctf, otf

def UpdateVolumeDisplay(volume, ctf=None, otf=None):
    if ctf is not None:
        # Update CTF
        volume._volume_property.set_color(ctf)
        volume._ctf = ctf
        volume.update_ctf = True

    if otf is not None:
        # Update OTF
        volume._otf = otf
        volume._volume_property.set_scalar_opacity(otf)

def showArray(array, isData):
    show3dArray(arrayTo3dArray(array), isData)

def show3dArray(ndArray, isData):
    vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(ndArray))
    if isData:
        ctf, otf = ImproveDataDisplay()
    else:
        ctf, otf = ImproveFilterDisplay()
    UpdateVolumeDisplay(vol, ctf, otf)

def Main(outputFolder, isData, normalized = True):
    assert isinstance(outputFolder, str)
    min, max = 0., 1.
    if os.path.splitext(args.file)[1] == '.arff':
        datasets, targets, targetMap = loadFromArff(args.file)
    elif os.path.splitext(args.file)[1] == '.npy':
        datasets = numpy.load(args.file)
        min = -1.
    else:
        assert False

    datasets = (lambda x : histogramEqualization(x, min=min, max=max) if normalized else x)(datasets)
    if normalized : assert (datasets.min(), datasets.max()) == (min, max)

    if not os.path.isdir("%s/Pictures" % outputFolder):
        os.makedirs("%s/Pictures" % outputFolder)

    global listIndex
    if listIndex is None or (len(listIndex) >= len(datasets)):
        listIndex = xrange(len(datasets))

    for index in listIndex:
        assert 0 <= index < len(datasets)
        mlab.figure("Index : %d" % index, bgcolor=(1,1,1))
        showArray(datasets[index], isData)
        mlab.savefig("%s/Pictures/Index_%d.png" % (outputFolder, index))
        if isData:
            saveData('%s/Pictures/%s_Index_%d.txt' % (outputFolder, targetMap.reverse_mapping[targets[index]], index), datasets[index])
        else:
            saveData('%s/Pictures/Index_%d.txt' % (outputFolder, index), datasets[index])
        mlab.close()

def Parser():
    global parser, args, listIndex
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--size', dest='size', nargs=len(tuple(defaultSize)), default=tuple(defaultSize), required=False)
    parser.add_argument('--range', dest='range', nargs=2, default=[0, 0], required=False)

    outputFolder = '../new_Outputs/ILC'

#    method = 'DenoisingAutoEncoder'
    method = 'RestrictedBoltzmannMachine'
    outputFolder += '/%s' % method
    filters = '%s/filters.npy' % outputFolder
    isData = False

#    data = '%s/test.arff' % outputFolder
#    isData = True

    args = parser.parse_args(['--range', '0', '500', filters])
    listIndex = xrange(int(args.range[0]), int(args.range[1]))
#    args = parser.parse_args([data])
    assert len(args.size) == len(defaultSize)
    args.size = tuple(args.size)
    Main(outputFolder, isData, normalized=True)

if __name__ == '__main__':
#    mlab.options.backend = 'envisage'
    Parser()
