#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

class ArffStruct(object):
    def __init__(self, relationName="", attributes=[], data=numpy.asarray([[]])):
        self.SetRelationName(relationName)
        self.SetAttributes(attributes)
        self.SetData(data)

    def GetRelationName(self):
        return self.__relationName
    def SetRelationName(self, new_relationName):
        assert isinstance(new_relationName, str)
        self.__relationName = new_relationName

    def GetAttributes(self):
        return self.__attributes
    def SetAttributes(self, new_attributes):
        assert isinstance(new_attributes, list)
        self.__attributes = new_attributes
    def AddNewAttributes(self, new_attribute):
        self.__attributes.append(new_attribute)

    def GetData(self):
        return self.__data
    def SetData(self, new_data):
        assert isinstance(new_data, numpy.ndarray) and len(new_data.shape) == 2
        self.__data = new_data