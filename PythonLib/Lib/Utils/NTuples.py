#! /usr/bin/env python
# -*- coding: utf-8 -*-

def NTuples(lst, n):
    if n == 1:
        return lst
    return zip(*[lst[i::n] for i in range(n)])