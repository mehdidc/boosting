#! /usr/bin/env python
# -*- coding: utf-8 -*-

class ArffAttribute(object):
    def __init__(self, name=""):
        self.SetName(name)

    def GetName(self):
        return self._name
    def SetName(self, new_name):
        assert isinstance(new_name, str)
        self._name = new_name

class ArffStringAttribute(ArffAttribute):
    def __init__(self, name=""):
        ArffAttribute.__init__(self, name)

class ArffNominalAttribute(ArffAttribute):
    def __init__(self, name="", nominalAttributes="{}"):
        ArffAttribute.__init__(self, name)
        self.__ParseString(nominalAttributes)

    def __ParseString(self, string):
        assert isinstance(string, str) and len(string) >= 2 and string[0] == '{' and string[-1] == '}'
        self.SetNominalAttributes()

    def GetNominalAttributes(self):
        return self.__