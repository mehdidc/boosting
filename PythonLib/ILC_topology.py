#! /usr/bin/env sage -python
# -*- coding: utf-8 -*-

import numpy
import os
import itertools
import multiprocessing
from math import cos, sin
try:
    import cPickle as pickle
except:
    import pickle

from sage.all import *

from Lib.Utils.NTuples import NTuples

def toTuple(x):
    return x if isinstance(x, tuple) else tuple((lambda y : [y] if not isinstance(y, list) else y)(x))

def isSubTuple(a, b):
    if not isinstance(a, tuple): a = toTuple(a)
    if not isinstance(b, tuple): b = toTuple(b)
    assert isinstance(a, tuple) and isinstance(b, tuple)
    return all([val in b for val in a])

def isPerfectSquare(n):
    assert isinstance(n, int) and n >= 0
    x = n // 2
    seen = set([])
    while x*x != n:
        x = (x + (n // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True

class GraphCell(object):
    def __init__(self, N=dict(), E=dict()):
        assert isinstance(N, dict) and isinstance(E, dict)
        self.N = N
        self.E = E

    def GetAllNodes(self):
        result = Set([])
        for set in self.N.values():
            result = result.union(Set(set))
        return result

    def GetAllLinks(self):
        result = Set([])
        for set in self.E.values():
            result = result.union(Set(set))
        return result

    def GetNodes(self, k):
        assert isinstance(k, int) and k >= 0
        return Set([uk for uk in self.GetAllNodes() if len(toTuple(uk)) == k + 1])

    def GetLinks(self, k):
        assert isinstance(k, int) and k >= 0
        return Set([ek for ek in self.GetAllLinks() if len(toTuple(ek[0])) == k + 1])

class Function(object):
    def __init__(self, f):
        assert isinstance(f, numpy.ndarray) and len(f.shape) == 1
        self.f = f

    def Get(self, x):
        return numpy.max([self.f[val] for val in toTuple(x)])

class Complex(object):
    def __init__(self, domain=(1,1,1)):
        assert isinstance(domain, tuple) and len(domain) == 3 and all([domain[n] > 0 for n in xrange(3)])
        self.domain = domain
        self.obj = dict()

    def ComputeGenericComplex(self):
        n = len(self.domain)
        genericComplex = dict()
        genericComplex[0] = list(xrange(numpy.prod(self.domain)))
        tmp = []
        Width, Height, Depth = self.domain
        for k in xrange(Depth):
            for i in xrange(Height):
                for j in xrange(Width):
                    tmp += sorted(self.__ComputeEdges(i, j, k))
        genericComplex[1] = sorted(list(set([tuple(sorted(x)) for x in tmp])))
        for i in list(xrange(2, n + 1, 1)):
            genericComplex[i] = self.__ComputeCell(i, genericComplex[i - 2], genericComplex[i - 1])
        for key, val in genericComplex.items():
            genericComplex[key] = Set(val)
        self.obj = genericComplex

    def __ComputeEdges(self, i, j, k):
        neighbors = self.__GetNeighbors(i, j, k)
        Width, Height, Depth = self.domain
        edges = []
        for neighbor in neighbors:
            edges.append([j + Width * (i + Height * k), neighbor])
        return edges

    def __ComputeCell(self, n, n_minus_2_cells, n_minus_1_cells):
        assert n >= 2
        n_minus_2_cells = [toTuple(x) for x in n_minus_2_cells]
        n_minus_1_cells = [toTuple(x) for x in n_minus_1_cells]
        n_minus_1_cellsUsed = []
        n_cells = []
        for n_minus_2_cell in n_minus_2_cells:
            V_n_minus_2_cells = [s for s in n_minus_1_cells if isSubTuple(n_minus_2_cell, s) and s not in n_minus_1_cellsUsed]
            if len(V_n_minus_2_cells) % 2**(n-1) != 0:
                V_n_minus_2_cells += V_n_minus_2_cells[:n-1]
#            print '%d-cell'%(n-2), n_minus_2_cell
            tmp1 = dict()
            for couples in NTuples(V_n_minus_2_cells, 2**(n-1)):
                assert all([len(couple) == 2**(n-1) for couple in couples])
#                print '  %d-cell existentes'%(n-1), couples
                diff = []
                for couple in couples:
                    diff += [x for x in couple if x not in diff]
                diff = NTuples([x for x in diff if not all([x in couple for couple in couples])], 2**(n-2))
#                print '    difference', diff
                for n_minus_1_cell in [s for s in n_minus_1_cells if s not in n_minus_1_cellsUsed and s not in couples]:
                    for d in [d for d in diff if isSubTuple(toTuple(d), n_minus_1_cell)]:
                        assert len(toTuple(d)) == 2**(n-2)
                        if not tmp1.has_key(toTuple(d)):
                            tmp1[toTuple(d)] = []
                        tmp1[toTuple(d)] += [toTuple(x) for x in toTuple(n_minus_1_cell) if x not in toTuple(d)]
                for key,val in tmp1.items():
                    tmp1[key] = list(set(val))
#                print '      %d-cell incidentes'%(n-2), tmp1
            tmp2 = []
            for permutation in list(itertools.permutations(tmp1.keys(), 2)):
                tmp2 += list(Set(tmp1[permutation[0]]).intersection(Set(tmp1[permutation[1]])))
            tmp2 = list(set(tmp2))
#            print '  %d-cell candidates'%(n-2), tmp2
            tmp3 = []
            for t in tmp2:
                tmp3 += [tuple(sorted([n_minus_2_cell] + [key for key, val in tmp1.items() if t in val] + [t]))]
            tmp4 = []
            for t in tmp3:
                s = ()
                for a in t:
                    s += toTuple(a)
                tmp4 += [toTuple(s)]
            n_minus_1_cellsUsed = list(set(n_minus_1_cellsUsed + V_n_minus_2_cells))
#            print '  %d-cell obtenues'%n, tmp4
            n_cells += tmp4
        res = []
        for r in NTuples(sorted(n_cells), 2**(n-2)):
            u = []
            for elt in r:
                u += list(Set(toTuple(elt)).union(Set(list(u))))
            res += [toTuple(list(set(u)))]
        assert all([len(x) == 2**n for x in sorted(res)])
        return sorted(res)

    def _isNext(self, current, next, axis):
        assert isinstance(axis, tuple) and len(axis) == len(self.domain) and \
               all([isinstance(val, int) and val >= 0 for val in axis]) and numpy.sum(axis) == 1
        if isinstance(current, tuple):
            assert len(current) == 3
            i,j,k = current
        elif isinstance(current, int):
            assert 0 <= current < numpy.prod(self.domain)
        Width, Height, Depth = self.domain
        return next == (j + axis[0]) + Width * ((i + axis[1]) + Height * (k + axis[2]))

    def __GetNeighbors(self, i, j, k):
        Width, Height, Depth = self.domain
        neighbors = []
        current = j + Width * (i + Height * k)
        for a in [-1,0,1]:
            for b in [-1,0,1]:
                for c in [-1,0,1]:
                    if (k + a) not in xrange(Depth) or (i + b) not in xrange(Height) or (j + c) not in xrange(Width):
                        neighbors += [None]
                    else:
                        neighbors += [(j + c) + Width * ((i + b) + Height * (k + a))]
        neighbors = sorted([x for x in neighbors if x is not None and x > current])
        tmp = []
        for neighbor in neighbors:
            if self._isNext((i,j,k), neighbor, (1,0,0)) or self._isNext((i,j,k), neighbor, (0,1,0)) or self._isNext((i,j,k), neighbor, (0,0,1)):
                tmp += [neighbor]
        neighbors = tmp
        return neighbors

    def GetCell(self, n):
        assert 0 <= n <= self.domain
        return Set(self.obj.values()[n])

    def GraphCell(self):
        N = dict()
        E = dict()
        for k in xrange(len(self.domain) + 1):
            N[k] = self.GetCell(k)
        for k in xrange(1, len(self.domain) + 1):
            tmp = []
            for uk in N[k - 1]:
                for wk_plus_1 in N[k]:
                    if isSubTuple(uk, wk_plus_1):
                        tmp += [(uk, wk_plus_1)]
            E[k - 1] = Set(tmp)
        return GraphCell(N=N, E=E)

    def Dump(self, filename):
        assert isinstance(filename, str)
        output = open(filename, 'w')
        obj = dict()
        for key, val in self.obj.items():
            obj[key] = list(val)
        pickle.dump(obj, output)
        output.close()

    def Load(self, filename):
        assert isinstance(filename, str) and os.path.exists(filename)
        pkl_file = open(filename)
        obj = pickle.load(pkl_file)
        self.obj = dict()
        for key, val in obj.items():
            self.obj[key] = Set(val)
        pkl_file.close()

class Queue(object):
    def __init__(self):
        self.data = []

    def pop_front(self):
        assert len(self.data) > 0
        res = self.data[0]
        self.data = self.data[0:]
        return res

    def isEmpty(self):
        return len(self.data) == 0

    def order(self, function):
        self.data = function(self.data)

    def add(self, d):
        self.data.append(d)

    def remove(self, d):
        self.data.remove(d)

class ComputeMorseSmaleComplex(object):
    def __init__(self, omega, g, complex, pertubateFunctionLevel=None):
        assert isinstance(omega, tuple) and len(omega) == 3
        self.omega = omega
        assert isinstance(g, numpy.ndarray) and g.size == numpy.prod(omega)
        self.g = Function(g)
#        if pertubateFunctionLevel is not None:
#            self.__PertubateFunction(pertubateFunctionLevel)
#        self.__CheckUnicity()
        assert isinstance(complex, Complex)
        self.complex = complex
        
    def __CheckUnicity(self):
        return self.g.size == len(set(self.g.flatten()))
        
    def __PertubateFunction(self, pertubateFunctionLevel):
        assert isinstance(self.omega, tuple) and len(self.omega) == 3
        I, J, K = self.omega
        for k in xrange(K):
            for i in xrange(I):
                for j in xrange(J):
                    self.g[j + J * (i + I * k)] += pertubateFunctionLevel * (float(i + I*j + I*J*k) / float(3*I*J*K))
    
    def __GetLowerStar(self, x):
        assert isinstance(x, int) and 0 <= x < numpy.prod(self.omega)
        L = dict()
        for key, val in self.complex.obj.items():
            for cell in val:
                if isSubTuple(x, cell) and self.g[x] == numpy.max([self.g[y] for y in toTuple(cell)]):
                    if not L.has_key(key):
                        L[key] = []
                    L[key] += [cell]
        return L

    def __G(self, cell):
        tmp = [(x, self.g[x]) for x in toTuple(cell)]
        return [x for x, val in sorted(tmp, key=lambda x : x[1])]

    def __GetFaces(self, beta):
        assert len(toTuple(beta)) > 1
        alpha_set = self.complex.obj[len(toTuple(beta)) - 1]
        faces = []
        for alpha in alpha_set:
            if isSubTuple(alpha, beta):
                faces += [alpha]
        return faces

    def __num_unpaired_faces(self, x, alpha, C, V):
        faces = self.__GetFaces(alpha)
        L = self.__GetLowerStar(x)
        res = 0
        cells = []
        for cell in L.values():
            if cell in faces and cell not in C or V:
                res += 1
                cells += [cell]
        return res, cells

    def __ProcessLowerStars(self):
        def orderList(l):
            def fun(x, g):
                if len(toTuple(x)) == 1:
                    return x, g[x]
                else:
                    max = (-1, float('Inf'))
                    for y in toTuple(x):
                        if (y, g[y]) >= max:
                            max = (y, g[y])
                    assert max[0] >= 0
                    return max
            tmp = [fun(x, self.g) for x in l]
            return [x[0] for x in sorted(tmp, key=lambda x : x[1])]
        C = {}
        V = dict()
        PQzero, PQone = Queue(), Queue()
        for x in xrange(numpy.prod(self.omega)):
            L_x = self.__GetLowerStar(x)
            if len(L_x.values()) == 1 and L_x.values()[0] == x:
                if C[len(toTuple(x)) - 1] is None:
                    C[len(toTuple(x)) - 1] = []
                C[len(toTuple(x)) - 1].append(x)
            else:
                g_delta = (float('Inf'), float('Inf'))
                delta = None
                for cell_1 in L_x[1]:
                    tmp = self.__G(cell_1)
                    if tmp <= g_delta:
                        delta = cell_1
                        g_delta = tmp
                assert delta is not None
                V[x] = delta
                for cell_1 in [c for c in L_x[1] if c != delta]:
                    PQzero.add(cell_1)
                PQzero.order(orderList)
                for cell in [c for c in L_x.values() if delta in self.__GetFaces(c) and self.__num_unpaired_faces(x, delta, C, V) == 1]:
                    PQone.add(cell)
                PQone.order(orderList)
                while not PQzero.isEmpty() or not PQone.isEmpty():
                    while not PQone.isEmpty():
                        alpha = PQone.pop_front()
                        num_unpaired_faces, pair = self.__num_unpaired_faces(x, alpha, C, V)
                        if num_unpaired_faces == 0:
                            PQzero.add(alpha)
                            PQzero.order(orderList)
                        else:
                            V[pair] = alpha
                            PQzero.remove(pair)
                            for beta in [c for c in L_x.values() if (alpha in self.__GetFaces(c) or pair in self.__GetFaces(c)) and self.__num_unpaired_faces(x,c,C,V)[0] == 1]:
                                PQone.add(beta)
                    if PQzero.isEmpty():
                        omega = PQzero.pop_front()
                        if C[len(toTuple(omega)) - 1] is None:
                            C[len(toTuple(omega)) - 1] = []
                        C[len(toTuple(omega)) - 1].append(omega)
                        for alpha in [c for c in L_x.values() if omega in self.__GetFaces(c) and self.__num_unpaired_faces(x,c,C,V)[0] == 1]:
                            PQone.add(alpha)
        return C, V

    # Algorithme 2
    def __AlternatingRestrictedBFS(self, G, V, R, cp):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                R.issubset(G.E) and\
                cp in G.N.list()

        # Initialize
        T = Set([]) # set of integrated links
        Q = multiprocessing.Queue()

        Q.put((cp, False)) # the links of the start node are unmatched

        # breadth-first search
        while not Q.empty():
            (up, flag) = Q.get() # get the next node and the flag of its links
            W = self.__AlternatingEdges(G, V, up, flag) # get the correctly flagged links
            W = W.intersection(R).difference(T) # apply restriction R and remove visited links T
            for (up, wk) in W.list():
                T = T.union(Set([(up, wk)])) # add links to the output
                Q.put((wk, not flag))

        assert T.issubset(R)
        return T

    # Algorithme 3
    def __AlternatingEdges(self, G, V, up, flag):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                up in G.N.list() and\
                isinstance(flag, bool)

        W = Set([]) # set of links

        if flag:
            links = Set([(a, wk) for (a, wk) in V.list() if a == up]) # links belong to V
        else:
            links = Set([(a, wk) for (a, wk) in G.E.difference(V).list() if a == up]) # links belong to E \ V

        assert isinstance(links, Set)
        W = Set([link for link in G.E.list() if link in links.list()])

        assert W.issubset(G.E)
        return W

    # Algorithme 4
    def __GetIntersection(self, G, V, S, l, j):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                S.issubset(G.E) and\
                l in [0,1,2] and\
                j in [0,1]

        C_S = Set([]) # TODO : critical nodes in the boundary of S
        I = Set([]) # initialize intersection of two manifolds

        # for each boundary critical node
        for cp in C_S.list():
            S = S.difference(I) # remove already visited links from the restriction S
            I = I.union(self.__AlternatingRestrictedBFS(G, V, S, cp)) # apply back-integration

        assert I.issubset(G.E)
        return I

    # Algorithme 5
    def __CombinatorialGradient(self, G, f):
        assert  isinstance(G, GraphCell) and\
                isinstance(f, Function)

        def ChooseLink(L):
            return Set([]) # TODO

        d = len(self.omega)
        V = Set([]) # initialize combinatorial gradient field

        for v0 in G.GetNodes(0).list(): # for each 0-node
            S = Set([v0]) # create its lower star
            W = Set([w for w in G.GetAllNodes().list() if f.Get(w) <= f.Get(v0)])
            for p in xrange(d):
                tmp = Set([])
                for up in [up for up in S.list() if len(toTuple(up)) == p]:
                    tmp = tmp.union(Set([wp_plus_1 for wp_plus_1 in W.list() if (up, wp_plus_1) in G.GetAllLinks()]))
                S = S.union(tmp)
            K = Set([elt[1] for elt in G.GetAllLinks() if elt[0] in S.list()]) # get the links connecting the lower star nodes
            C = Set([]) # initialize list of flagged nodes
            abort = False
            while not abort: # start homotopic expansion
                abort = True
                p = 0
                while not abort and p <= d - 1: # for each dimension
                    C_union_N_V = C.union(Set([link[0] for link in V.list()]))
                    T = Set([(up, wp_plus_1) for (up, wp_plus_1) in K.list() if up and wp_plus_1 not in C_union_N_V.list()]) # nodes are not covered
                    L = T.difference(Set([])) # and unique
                    if L.cardinality() != 0: # there are valid expansions
                        V = V.union(ChooseLink(L)) # apply homotopic expansion
                        abort = False # continue homotopic expansion
                    else: # there is no valid expansions
                        k = 0
                        while not abort and k <= d: # for each dimension
                            unflagged_nodes = Set([])
                            if unflagged_nodes.cardinality() != 0:
                                C = C.union(Set([unflagged_nodes.list()[0]])) # flag an arbitrary unflagged node
                                abort = False # continue homotopic expansion
                            k += 1
                    p += 1

        assert V.issubset(G.GetAllLinks())
        return V

    # Algorithme 7
    def __ComputeBoundaryMatrix(self, G, V, j, l):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                j in [0,1] and\
                l in [0,1,2]

        delta_l_plus_1 = {} # boundary matrix

        # integrate all manifolds
        S = self.__GetAllManifolds(G, V, l, j)

        # compute intersection w.r.t. the boundary nodes
        I = self.__GetIntersections(G, V, S, l, j)

        assert self.C is not None
        # for all critical nodes of index l + 1 − j
        for cp in self.C[l + 1 - j]:
            Cc = self.__CountPaths(G, V, I, cp, l) # compute boundary nodes Cc w.r.t. Z2
            for wk in Cc.list():
                p, k = len(toTuple(cp)), len(toTuple(wk))
                # add node to ∂l+1
                if k < p: # the node is a boundary node
                    if delta_l_plus_1[(cp,wk)] is None:
                        delta_l_plus_1[(cp,wk)] = 1
                    else:
                        delta_l_plus_1[(cp,wk)] += 1
                else: # the node is a coboundary node
                    if delta_l_plus_1[(wk,cp)] is None:
                        delta_l_plus_1[(wk,cp)] = 1
                    else:
                        delta_l_plus_1[(wk,cp)] += 1

        return delta_l_plus_1

    # Algorithme 8
    def __GetAllManifolds(self, G, V, l, j):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                l in [0,1,2]

        E_l = G.GetLinks(l) # links of index l
        S = Set([]) # initialize S : set of link representing all manifolds

        assert self.C is not None
        # for all critical nodes of index l + 1 − j
        for cp in self.C[l + 1 - j]:
            E_l = E_l.difference(S) # remove all already visited links
            S = S.union(self.__AlternatingRestrictedBFS(G, V, E_l, cp)) # add current manifold to S

        assert S.issubset(G.E)
        return S

    # Algorithme 9
    def __CountPaths(self, G, V, I, cp, l):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                I.issubset(G.E) and\
                cp in G.N.list()

        Cc = Set([]) # adjacent critical nodes w.r.t. Z2
        N_S = self.__GetManifoldNodes(G, V, I, cp, l) # nodes covered by 1-separatices of cp
        C_NS = Set([up for up in N_S.list() if all([a != up for (a,wk) in V.list()])]) # critical nodes in N_S
        P = Set([]) # control container for the breadth-first search
        L = Set([cp]) # initialize list of visited nodes with cp
        Q = multiprocessing.Queue()
        Q.put((cp, False)) # all links of cp are unmatched
        while not Q.empty(): # constrained breadth-first search
            (up, flag) = Q.get() # get the next node and the flag of its links
            P = P.union(Set([up])) # add the node to the control container
            W = self.__AlternatingEdges(G, V, up, flag) # get the correctly flagged links
            W = W.intersection(I) # consider only links that are also part of the intersection I
            for (up, wk) in W.list(): # for each link={start node, end node}
                if wk in N_S.list(): # the end node must be covered by the 1-separatrices of cp
                    if up in L.list(): # if the start node is flagged
                        L = L.symmetric_difference(Set([wk])) # flag the end node w.r.t. Z2
                    Z = self.__AlternatingEdges(G, V, wk, flag) # get the links of the end node
                    Z = Z.intersection(I) # restrict them to the intersection I
                    N_Z = Set([up for up in G.N.list() if not all([a != up for (a,wk) in Z.list()])]) # get their start nodes
                    if N_Z.issubset(P): # all start nodes must already be processed
                        Q.put((wk, not flag))
        Cc = L.intersection(C_NS).difference(Set([cp])) # restrict visited nodes to the critical nodes except cp

        assert Cc.issubset(G.N)
        return Cc

    # Algorithme 10
    def __GetManifoldNodes(self, G, V, I, cp, l):
        assert  isinstance(G, GraphCell) and\
                V.issubset(G.E) and\
                cp in G.N.list() and\
                l in [0,1,2]

        E_l = G.GetLinks(l) # links of index l
        S = self.__AlternatingRestrictedBFS(G, V, E_l.intersection(I), cp) # integration restricted to I
        N_S = Set([]) # initialize set of nodes covered by the integrated manifold

        for (up, wk) in S.list(): # for each link={start node, end node}
            if up not in N_S.list(): # start node was not yet added
                N_S = N_S.union(Set([up])) # add start node
            if wk not in N_S.list(): # end node was not yet added
                N_S = N_S.union(Set([wk])) # add end node

    def Run(self):
        G = self.complex.GraphCell()
        V = self.__CombinatorialGradient(G, self.g)
#        self.C, V = self.__ProcessLowerStars()
#        delta_1 = self.__ComputeBoundaryMatrix(G, V, 0, 0)
#        delta_2 = self.__ComputeBoundaryMatrix(G, V, 0, 1)
#        delta_3 = self.__ComputeBoundaryMatrix(G, V, 1, 2)
#        print delta_1
    
def test_ComputeCubicalComplex():
    D = (18, 18, 30)
    c = Complex(D)
    c.ComputeGenericComplex()
    c.Dump('complex.sobj')

def test_ComputeMorseSmaleComplex():
    domain = (4, 4, 4)
    W, H, D = domain
    g = numpy.zeros(shape=(numpy.prod(domain),))
    for x in xrange(-2, 3):
        for y in xrange(-2, 3):
            for z in xrange(-2, 3):
                g[y + W * (x + H * z)] = 1*sin(1*x)*sin(2*y)*sin(3*z)\
                           + 2*sin(2*x)*sin(1*y)*sin(3*z)\
                           + 3*sin(3*x)*sin(2*y)*sin(1*z)\
                           + 4*sin(1*x)*sin(3*y)*sin(2*z)\
                           + 5*sin(2*x)*sin(3*y)*sin(1*z)\
                           + 6*sin(3*x)*sin(1*y)*sin(2*z)\
                           + 1*cos(3*x)*cos(1*y)*cos(2*z)\
                           + 2*cos(2*x)*cos(1*y)*cos(3*z)\
                           + 3*cos(1*x)*cos(2*y)*cos(3*z)\
                           + 4*cos(3*x)*cos(2*y)*cos(1*z)\
                           + 5*cos(2*x)*cos(3*y)*cos(1*z)\
                           + 6*cos(1*x)*cos(3*y)*cos(2*z)
    C = Complex(domain)
    C.ComputeGenericComplex()
    method = ComputeMorseSmaleComplex(domain, g, C)
    method.Run()

def main():
    test_ComputeMorseSmaleComplex()

if __name__ == '__main__':
    main()