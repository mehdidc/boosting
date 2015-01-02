#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from ArffStruct import ArffStruct

class ArffParser(object):
    def __init__(self, filename, isSparse=False):
        self.SetFilename(filename)
        self.SetSparsity(isSparse)

    def GetFilename(self):
        return self.__filename
    def SetFilename(self, new_filename):
        assert isinstance(new_filename, str) and os.path.exists(new_filename) and os.path.splitext(new_filename)[1] == ".arff"
        self.__filename = new_filename

    def IsSparse(self):
        return self.__isSparse
    def SetSparsity(self, isSparse):
        assert isinstance(isSparse, bool)
        self.__isSparse = isSparse
    def ToogleSparsity(self):
        self.__isSparse = not self.__isSparse

    def Run(self):
        arffStruct = ArffStruct()
        lines = self.__Load()
        while True:
            line = lines[0]
            if line[0] == '@':
                if line.upper() == "@DATA":
                    self.__ParseData(lines[1:], arffStruct)
                    break
                self.__ParseHeader(line, arffStruct)
            else:
                assert False
            lines = lines[1:]
            if len(lines) <= 0:
                break
        print arffStruct.GetAttributes()
        return arffStruct

    def __Load(self):
        return [line.strip() for line in open(self.GetFilename()).readlines() if len(line.strip()) > 0]

    def __ParseHeader(self, line, arffStruct):
        words = line.split()
        if len(words) == 2 and words[0].upper() == '@RELATION':
            arffStruct.SetRelationName(words[1])
        elif len(words) == 3 and words[0].upper() == '@ATTRIBUTE':
            arffStruct.AddNewAttributes({'name' : words[1], 'type' : words[2]})
        elif len(words) > 3 and words[2][0] == '{':
            new_line = ' '.join([words[0], words[1], ''.join(words[2:])])
            self.__ParseHeader(new_line, arffStruct)
        else:
            assert False

    def __ParseData(self, lines, arffStruct):
        while True:
            line = lines[0]
            lines = lines[1:]
            if len(lines) <= 0:
                break

def main():
    filename = '../../../../Databases/ILC_DeepLearning_Experiment/Deep/PixelEnergy_2GeV_e6151_130207.arff'
    arffParser = ArffParser(filename)
    arffParser.Run()

if __name__ == '__main__':
    main()