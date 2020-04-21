from scipy import sparse

import scipy.special
import numpy as np
from itertools import combinations
import math
import time
from scipy.sparse import coo_matrix, diags
import numpy_indexed as npi
from scipy.sparse import identity
from scipy.sparse.linalg import eigs, eigsh


class Basis:

    def __init__(self, M, N, T, U, lat):
        """ M= # of sites on the lattice
            N= # of bosons on the lattice
            D= dim(HilbertSpace)
            lat= lattice parameter defining lat X lat square lattice
            U= interaction strength
            T=hopping strength
        """

        self.M = M
        self.N = N
        self.lat = lat
        self.U = U
        self.T = T

    def __init__(self, M, N, T, U, lat=None):
        """ M= # of sites on the lattice
            N= # of bosons on the lattice
            D= dim(HilbertSpace)
            lat= lattice parameter defining lat X lat square lattice
        """
        self.M = M
        self.N = N
        self.lat = lat
        self.U = U
        self.T = T

    def D(self):

        """Computes dimension of the Hilbert space"""

        return int(scipy.special.binom(self.N + self.M - 1, self.N))

    def occuNum(self):

        """Generating all possible occupation number configurations in a generator format"""

        for c in combinations(range(self.N + self.M - 1), self.M - 1):
            yield np.flip([b - a - 1 for a, b in zip((-1,) + c, c + (self.N + self.M - 1,))])

    def basisArray(self):

        """Generates D X M size of array storing all basis states"""

        states = np.zeros((self.D(), self.M), dtype=int)
        count = 0
        generator = self.occuNum()

        for row in states:
            item = next(generator)
            row[:] = item
            count += 1

        return states[:count, :]


class Hamiltonian(Basis):

    def __init__(self, M, N, T, U, lat):
        super().__init__(M, N, T, U, lat)
        self.states = self.basisArray()

    def __init__(self, M, N, T, U, lat=1):
        super().__init__(M, N, T, U, lat)

    def H_int(self):
        """ Generates sparse diagonal interaction matrix with interaction strength U"""

        diag = [0.5 * self.U * sum(row * row - row) for row in self.occuNum()]
        return diags(diag, 0)

    def lattice2D(self):
        """Construction of hopping terms for 2D square lattice"""

        lat = self.lat
        verHopList = [[i, i + lat] for i in range(self.M - lat)]
        horHopList = [[i, i + 1] for i in range(self.M) if (i + 1) % lat != 0]
        verHopListConj = [[i + lat, i] for i in range(self.M - lat)]
        horHopListConj = [[i + 1, i] for i in range(self.M) if (i + 1) % lat != 0]
        hopp_link_sqr = verHopList + verHopListConj + horHopList + horHopListConj

        return hopp_link_sqr

    def lattice1D(self):
        """Construction of hopping terms for 1D lattice"""

        links = [[i, (i + 1)] for i in range(self.M - 1)]
        links_conj = [[i + 1, i] for i in range(self.M - 1)]
        hopp_link = links + links_conj

        return hopp_link

    def mat_arg(self, lat_link):
        """ lat_link = whether 1D system or 2D system
            Its argument comes from latticeND() function
            It generates non-zero positions of sparse matrix with amplitudes
        """
        A = self.basisArray()
        for j in range(A.shape[0]):
            for sublink in lat_link:
                if A[j, sublink[0]] >= 1:
                    hoppedVec = np.array(A[j]).tolist()
                    hoppedVec[sublink[0]] -= 1
                    hoppedVec[sublink[1]] += 1
                    i = int(npi.indices(A, np.array([hoppedVec]), missing='mask'))
                    amp = round(math.sqrt((A[j, sublink[1]] + 1) * A[j, sublink[0]]) * -self.T, 3)
                    yield [i, j, amp]

    def hoppedStates(self, lat_link):
        """Generates non-zero-amplitude basis states after hopping terms applied
           Its argument comes from latticeND() function
        """

        A = self.basisArray()
        for j in range(A.shape[0]):
            for sublink in lat_link:
                if A[j, sublink[0]] >= 1:
                    hoppedVec = np.array(A[j]).tolist()
                    hoppedVec[sublink[0]] -= 1
                    hoppedVec[sublink[1]] += 1
                    yield hoppedVec

    def H_kin(self, latt):
        """Sparse kinetic Hamiltonian is generated"""
        row, col, data = [], [], []

        for p in self.mat_arg(latt):
            row.append(p[0])
            col.append(p[1])
            data.append(p[2])

        a = coo_matrix((data, (row, col)), shape=(self.D(), self.D()))

        return a


"""start_time = time.time()
basis = Basis(2, 2, 1, 1)"""

# MAIN PROGRAM
"""h = Hamiltonian(4, 4, 1, 1)
A = h.H_int()
B = h.H_kin(h.lattice1D())
D = (A.tocsr() + B.tocsr())
#vals, vecs = sp.sparse..eigsh(D, k=3)
#print(vals)

A = identity(10, format='csc')
A.setdiag(range(1, 11))

vals,vecs = eigs(D, 20, sigma=0)

valss, vecss = eigsh(D, k=10)
print(vals)
print()
print(valss)


#print("--- %s seconds ---" % (time.time() - start_time))

# print(A[:1, ])  # first row, all columns
#  print(A[:,2])  # all rows, second column"""
