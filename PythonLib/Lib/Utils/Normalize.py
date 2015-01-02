#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import multiprocessing
from functools import partial

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
    print ndArray.shape
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
