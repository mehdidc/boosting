#! /usr/bin/env python
# -*- coding: utf-8 -*-

from .. import *
from Enums import *
from Normalize import *

class Data(object):
    __trainSet = []
    __validationSet = []
    __testSet = []
    __trainTargets = []
    __validationTargets = []
    __testTargets = []
    __normalizeSize = []
    __offset = 0
    __grayScale = False

    def __init__(self, normalizeSize, offset=0, trainSet=[], validationSet=[], testSet=[], trainTargets=[], validationTargets=[], testTargets=[], grayScale=False, negate=False):
        if offset >= 0:
            self.__offset = offset
        else:
            raise Exception("Data Init: Incorrect Offset")
        
        if isinstance(normalizeSize, tuple):
            i = 0
            for i in xrange(len(normalizeSize)):
                if normalizeSize[i] <= 0:
                    break
            if i < len(normalizeSize)-1:
                raise Exception("Data Init : Incorrect Normalization normalizeSize")
            else:
                self.__normalizeSize = tuple([x - (offset * 2) for x in normalizeSize])
        else:
            self.__normalizeSize = normalizeSize - (offset * 2)

        if len(trainTargets) == len(trainSet):
            self.__trainSet = trainSet
            self.__trainTargets = trainTargets
        else:
            raise Exception("Incorrect trainSet/trainTargets")
        if len(validationTargets) == len(validationSet):
            self.__validationSet = validationSet
            self.__validationTargets = validationTargets
        else:
            raise Exception("Incorrect validationSet/validationTargets")
        if len(testTargets) == len(testSet):
            self.__testSet = testSet
            self.__testTargets = testTargets
        else:
            raise Exception("Incorrect testSet/testargets")

        self.__grayScale = grayScale
        self.__negate = negate

    def addUniqueData(self, filename, inputType, normalized=False):
        import os
        if not os.path.isfile(filename):
            raise Exception("addUniqueData : \"" + filename + "\" is not a file")
        if inputType >= DataCategory.TRAIN and inputType <= DataCategory.TEST:
            im = Normalize(self.Size(), filename, offset=self.__offset)
            array = []
            import ImageOps
            if self.__grayScale:
                im = ImageOps.grayscale(im)
                if self.__negate:
                    im = ImageOps.invert(im)
                if normalized:
                   array = [float(x) / 255. for x in list(im.getdata())]
                else:
                   array = [x for x in list(im.getdata())]
            else:
                for p in list(im.getdata()):
#                    array.append(zip([float(x)/255. for x in p]))
                    for i in xrange(len(p)):
                        if normalized:
                            array.append(float(p[i]) / 255.)
                        else:
                            array.append(p[i])
                    array = [x for x in array]
            self.Inputs()[inputType].append(array)
            import string
            groundTruth = string.split(os.path.splitext(os.path.split(filename)[1])[0], '-')[0]
            self.Targets()[inputType].append(int(groundTruth))
        else:
            raise Exception("Incorrect DataCategory")

    def addMultipleData(self, dirname, inputType, normalized=False):
        import os#, sys
        for root, dirs, files in os.walk(dirname):
            for i in files:
#                sys.stdout.write('.')
                if os.path.splitext(i)[1] == "":
                    continue
                self.addUniqueData(os.path.join(root, i), inputType, normalized)
#        sys.stdout.write('\n')

    def setTestFromTrain(self, ratio=1.):
        if len(self.__testSet) != 0:
            pass
        else:
            import random
            if ratio >= 0 and ratio <= 1:
                if ratio != 1.:
                    trainSet = []
                    trainTargets = []
                    testSet = []
                    testTargets = []
                    p = ratio
                    i = 0
                    for x in self.Inputs()[DataCategory.TRAIN]:
                        p = (ratio * len(self.Inputs()[DataCategory.TRAIN]) - len(trainSet)) / (len(self.Inputs()[DataCategory.TRAIN]) - i)
                        i += 1
                        if random.random() < p:
                            trainSet.append(x)
                            trainTargets.append(self.Targets()[DataCategory.TRAIN][i - 1])
                        else:
                            testSet.append(x)
                            testTargets.append(self.Targets()[DataCategory.TRAIN][i - 1])
                    self.__trainSet = trainSet
                    self.__trainTargets = trainTargets
                    self.__testSet = testSet
                    self.__testTargets = testTargets
            else:
                raise Exception("setTestFromTrain : Incorrect Ratio")

    def importFromARFF(self, filename, inputSet=DataCategory.TRAIN):
        import os
        import string
        assert os.path.exists(filename)
        inFile = open(filename)
        attributes = []
        dataFile = {}
        while 1:
            line = inFile.readline()
            if line[:9] == "@RELATION":
                tokens = string.split(line)
                assert len(tokens) <= 3
                if tokens[-1] == "\n":
                    del tokens[-1]
                dataFile['relation'] = tokens[1].strip()
            if line[:10] == "@ATTRIBUTE":
                tokens = string.split(line)
                attribute = {}
                attribute['name'] = tokens[1].strip()
                if tokens[1] == "class":
                    attribute['type'] = line[17:-1].strip()
                else:
                    attribute['type'] = tokens[2].strip()
                attributes.append(attribute)
            if line[:5] == "@DATA":
                break
        data = []
        size = 0
        for line in inFile:
            tokens = string.split(line,",")
            input = map(float,tokens[:-1])
            if len(input) != len(attributes)-1:
                print("len(point[" + str(size) + "]) == " + str(len(input)) +\
                          " != numOfAttributes == " + str(len(attributes)-1))
                exit(1)
            label = tokens[-1].strip()
            point = {}
            point['input'] = input
            point['label'] = label
            data.append(point)
            size += 1
        dataFile['attributes'] = attributes
        dataFile['data'] = data
        self.SetInputs([], inputSet)
        self.SetTargets([], inputSet)
        for p in data:
            self.Inputs()[inputSet].append(p['input'])
            self.Targets()[inputSet].append(p['label'])

    def export2ARFF(self, trainFilename="trainDatabase.arff", validationFilename="validationDatabase.arff", testFilename="testDatabase.arff", directory="./outputDir/"):
        trainFile = open(trainFilename, 'w')
        validationFile = open(validationFilename, 'w')
        testFile = open(testFilename, 'w')
        files = [trainFile, validationFile, testFile]

        import os
        if not os.path.exists(directory):
            os.system("mkdir -p " + directory)

        for inputType in (DataCategory.TRAIN, DataCategory.VALIDATION, DataCategory.TEST):
            files[inputType].write("@RELATION\t" + DataCategory.reverse_mapping[inputType] + "\n\n")
            size = 1
            for i in xrange(0, len(self.NormalizedSize()), 1):
                size *= self.NormalizedSize()[i]
            for i in xrange(0, size, 1):
                if self.__grayScale:
                    files[inputType].write("@ATTRIBUTE\t" + "pixel_" + str(i) + "\tNUMERIC" + "\n")
                else:
                    files[inputType].write("@ATTRIBUTE\t" + "pixel_" + str(i) + "_R" + "\tNUMERIC" + "\n")
                    files[inputType].write("@ATTRIBUTE\t" + "pixel_" + str(i) + "_G" + "\tNUMERIC" + "\n")
                    files[inputType].write("@ATTRIBUTE\t" + "pixel_" + str(i) + "_B" + "\tNUMERIC" + "\n")
            targetClass = ""
            for i in xrange(0, 9, 1):
                targetClass += str(i) + ","
            targetClass += str(9)
            files[inputType].write("@ATTRIBUTE\t" + "class" + "\t{" + targetClass + "}" + "\n")
            files[inputType].write("\n")

            string = "@DATA\n"
            for i in xrange(0, self.Length()[inputType], 1):
                im = self.Inputs()[inputType][i]
                for j in xrange(0, size, 1):
                    pixel = im[j]
                    if self.__grayScale:
                        string += str(pixel) + ","
                    else:
                        nb_channel = 3
                        for k in xrange(0, nb_channel, 1):
                            string += str(pixel[k]) + ","
                target = self.Targets()[inputType][i]
                string += str(target)
                string += "\n"
            files[inputType].write(string)

        trainFile.close()
        validationFile.close()
        testFile.close()

        os.system("mv " + trainFilename + ' ' + validationFilename + ' ' + testFilename + ' ' + directory)

    def Inputs(self):
        return self.__trainSet, self.__validationSet, self.__testSet
    def SetInputs(self, array, inputCategory):
        if inputCategory == DataCategory.TRAIN:
            self.__trainSet = array
        elif inputCategory == DataCategory.VALIDATION:
            self.__validationSet = array
        elif inputCategory == DataCategory.TEST:
            self.__testSet = array
    def Targets(self):
        return self.__trainTargets, self.__validationTargets, self.__testTargets
    def SetTargets(self, array, inputCategory):
        if inputCategory == DataCategory.TRAIN:
            self.__trainTargets = array
        elif inputCategory == DataCategory.VALIDATION:
            self.__validationTargets = array
        elif inputCategory == DataCategory.TEST:
            self.__testTargets = array
    def Size(self):
        return self.__normalizeSize
    def NormalizedSize(self):
        if isinstance(self.__normalizeSize, tuple):
            return tuple([x + (self.__offset * 2) for x in self.__normalizeSize])
        else:
            return (self.__normalizeSize + (self.__offset * 2),)
    def Length(self):
        return len(self.__trainSet), len(self.__validationSet), len(self.__testSet)
    def IsGrayScale(self):
        return self.__grayScale