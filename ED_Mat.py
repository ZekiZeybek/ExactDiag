from scipy import sparse as sp
import scipy.special
import numpy as np
import math
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from math import floor

#link(M) for generating kronecker product list contributed by github.com/hsaidc

# Defining global variables

M = 6  # number of sites on a given lattice
N = 2  # number of bosons 
l_ver = 2  # vertical length of the lattice
l_hor = 3  # horizontal length of the lattice
a = diags([math.sqrt(N) for N in range(1, N + 1)], 1)  # a
I = sp.identity(N + 1)  # spars.te identity matrix
t = 1
U = 0
a = sp.csr_matrix(a)
I = sp.csr_matrix(I)

def dimHil(M, N):
    """Computes dimension of the Hilbert space"""

    return int(scipy.special.binom(N + M - 1, N))


def opSize(M, N):

    return np.power((N + 1), M)


def link(M):
    identity_matrix = I
    random_matrix = a

    m = M - 1
    number_of_identity_matrix = m
    number_of_sub_lists = number_of_identity_matrix + 1
    total_number_of_elements = number_of_sub_lists * number_of_sub_lists

    link = []
    tmp = []

    for i in range(0, total_number_of_elements):
        row = floor(i / number_of_sub_lists)
        column = i % number_of_sub_lists
        if row == column:
            tmp.append(random_matrix)
        else:
            tmp.append(identity_matrix)
        if column == number_of_identity_matrix:
            link.append(tmp)
            tmp = []

    return link

def opGen(M):
    """ Generating many body creatin annihilation operators
        returns [....,all crea.,....,all annih.,....]
    """
    l=[]
    for sub in link(M):
        for i in range(M-1):
            sub[0] = sp.kron(sub[0], sub[i + 1])

        l.append(sub[0])
    l_T = [i.T for i in l]
    return l_T + l

def N_op():
    """Number operator corresponding to N particles"""

    N_op = sp.csr_matrix((opSize(M, N), opSize(M, N)))
    for i in range(M):
        N_op += opGen(M)[i] * opGen(M)[i + M] #M=4

    return N_op


def H_kin(a, I, t):
    """ !Operator form of kinetic hamiltonian
        !Must take hopTermGen() to form corresponding creation and annihilation operators
    """
    hopp_link = hopGen(M,l_ver,l_hor)
    H_kin = sp.csr_matrix((opSize(M,N),opSize(M,N)))
    for sub in hopp_link:
        H_kin += opGen(M)[sub[0]]*opGen(M)[sub[1] + M] # M=4

    return H_kin.dot(t)

def H_int():
    """Operator form of interaction hamiltonian"""
    H_int = sp.csr_matrix((opSize(M, N), opSize(M, N)))
    for i in range(M):
        H_int += opGen(M)[i] * opGen(M)[i + M]* opGen(M)[i] * opGen(M)[i + M]\
                 - opGen(M)[i] * opGen(M)[i + M]

    return H_int.dot(U * 0.5)


def H_all():
    """ """
    A = np.abs(np.diag(N_op().todense()) - N)
    iN = np.where(A < np.power(10, float(-6)))
    N_o = N_op().todense()
    iN_arr = np.array(iN)
    l = [iN_arr[0, i] for i in range(len(iN_arr[0]))]
    l = np.array(l)
    N_p = N_o[:, l] / N
    N_p = sp.csr_matrix(N_p)

    H_N = sp.csr_matrix.transpose(N_p)*(H_kin(a, I, t) + H_int())*N_p  # Projecting Hamiltonian into N=2 subspace

    return H_N



def Eig():
    """ Eigenvalues are returned """

    E1 = eigsh(H_all(), k=dimHil(M,N)-1, return_eigenvectors=False)

    return E1

def hopGen(M, l_ver, l_hor):
    "Generating relevant hopping connections for a given 2D lattice geometry"

    verHopList = [[i, i + l_ver] for i in range(M - l_ver)]
    horHopList = [[i, i + 1] for i in range(M) if (i + 1) % l_hor != 0]
    verHopListConj = [[i + l_ver, i] for i in range(M - l_ver)]
    horHopListConj = [[i + 1, i] for i in range(M) if (i + 1) % l_hor != 0]
    hopp_link_sqr = verHopList + verHopListConj + horHopList + horHopListConj

    return hopp_link_sqr


def hopGen1D(M):
    """Generating relevant hopping connections for a given 1D lattice geometry """

    links = [[i, (i + 1)] for i in range(M - 1)]
    links_conj = [[i + 1, i] for i in range(M - 1)]
    hopp_link = links + links_conj

    return hopp_link


print(Eig())
