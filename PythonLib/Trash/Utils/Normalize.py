#! /usr/bin/env python
# -*- coding: utf-8 -*-

import Image
import numpy

def Normalize(size, infile, offset=4):
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGBA', [x+(2*offset) for x in size], (255, 255, 255, 0))
    background.paste(
        im,
        ((size[0] - im.size[0] + (2*offset)) / 2, (size[1] - im.size[1] + (2*offset)) / 2))
    return background
    
def Normalize2dArray(array):
    assert isinstance(array, numpy.ndarray) and len(array.shape) == 2
    res = []
    for row in array.astype(float):
        tmp = (row.astype(float) - row.astype(float).min()) / (row.astype(float).max() - row.astype(float).min())
        assert tmp.min() == 0. and tmp.max() == 1.
        res.append(tmp)
    return numpy.asarray(res)