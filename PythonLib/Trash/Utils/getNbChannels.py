#! /usr/bin/env python
# -*- coding: utf-8 -*-

def getNbChannels(shape, dims):
    assert (len(shape) == dims or len(shape) == dims + 1)
    if len(shape) == dims:
        return shape, 1
    else:
        res = shape[dims]
        shape = shape[:dims]
        return shape, res