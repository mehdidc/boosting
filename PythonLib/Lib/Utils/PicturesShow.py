#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import Image

class PicturesShow(object):
    def __init__(self,
                 npArray=None,
                 imgShape=(28, 28),
                 nbChannels=1,
                 nbRows=1,
                 tileSpace=(1, 1),
                 backgroundColor=(0, 0, 0)
                 ):
        assert npArray is not None
        assert len(npArray.shape) == len(imgShape) == len(tileSpace) == 2
        assert numpy.prod(imgShape)*nbChannels == npArray.shape[1]
        assert divmod(npArray.shape[0], nbRows)[1] == 0
        for i in xrange(len(npArray.shape)):
            assert imgShape[i] >= 1
            assert tileSpace >= 1
        assert nbChannels >= 1
        assert nbRows >= 1
        assert len(backgroundColor) == 3
        for i in xrange(3):
            assert backgroundColor[i] in range(0, 255)
        
        self.image = None
        self.pictures = npArray
        self.imgShape = imgShape
        self.nbChannels = nbChannels
        self.grid = (nbRows, self.pictures.shape[0] / nbRows)
        self.tileSpace = tileSpace
        self.backgroundColor = ()
        for i in xrange(self.nbChannels):
            self.backgroundColor += (0,)
    
    def getImageFromData(self, imData, imageMode):
        from NTuples import NTuples
        im = Image.new(imageMode, self.imgShape)
        im.putdata(NTuples(imData, self.nbChannels))
        return im
    
    def ComputeFinalImage(self):
        if self.nbChannels == 1:
            imageMode = 'L'
        elif self.nbChannels == 3:
            imageMode = 'RGB'
        elif self.nbChannels == 4:
            imageMode = 'RGBA'
            self.backgroundColor = self.backgroundColor[:2] + (255,)
        else:
            raise Exception("PicturesShow : Incorrect nbChannels !!!")
            
        pictures = []
        for im in self.pictures:
            pictures.append(self.getImageFromData(im, imageMode))
            
        final_image_size = (self.grid[1] * (self.imgShape[1] + self.tileSpace[1]), self.grid[0] * (self.imgShape[0] + self.tileSpace[0]))
        final_image = Image.new(imageMode, final_image_size)
        offset = self.tileSpace
        pos = (0, 0)
        for im in pictures:
            if pos[1] >= self.grid[1]:
                pos = (pos[0] + 1, 0)
                offset = (self.tileSpace[0], offset[1] + self.imgShape[1] + self.tileSpace[1])
            final_image.paste(im, offset)
            pos = (pos[0], pos[1] + 1)
            offset = (offset[0] + self.imgShape[0] + self.tileSpace[0], offset[1])
        self.image = final_image
        return final_image
        
    def Show(self):
        if self.image is None:
            self.ComputeFinalImage()
        self.image.show()
        
    def Save(self, filename):
        if self.image is None:
            self.ComputeFinalImage()
        self.image.save(filename)
        
def test_PicturesShow(nbChannels):
    shape = (28, 28)
    import time
    numpy.random.seed(int(time.clock()))
    inputs = numpy.random.random_integers(low=0, high=255, size=(7000, numpy.prod(shape)*nbChannels))
    show = PicturesShow(inputs, imgShape=shape, nbRows=100, nbChannels=nbChannels)
    show.Show()

if __name__ == "__main__":
    test_PicturesShow(nbChannels=1)
    test_PicturesShow(nbChannels=3)