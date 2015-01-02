#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
import mdp
import numpy

def main():
    whiteningNode = mdp.nodes.WhiteningNode()
    pcaNode = mdp.nodes.PCANode()
    X = numpy.random.random((10, 10))
    print X
    whiteX = pcaNode.execute(X)
    print whiteX

if __name__ == '__main__':
    main()