#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ..Utils.Enums import enum

DataCategory = enum('TRAIN', 'VALIDATION', 'TEST')

TrainingMethod = enum('DAE', 'RBM', 'SDAE', 'DBN')

LossFunctionType = enum('SQUARE', 'CROSS_ENTROPY')
CorruptionType = enum('SALT_AND_PEPPER', 'GAUSSIAN_NOISE', 'BINOMIAL_NOISE')