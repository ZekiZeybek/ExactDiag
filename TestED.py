import ED
from scipy.sparse.linalg import eigs, eigsh

# MAIN PROGRAM
h = ED.Hamiltonian(4, 2, 1, 1,2)
h2 = ED.Hamiltonian(4,1,1,1,2)
A = h2.H_int()
B = h2.H_kin(h2.lattice2D())
D = (A.tocsr() + B.tocsr())

print(h2.lattice2D())

#vals,vecs = eigs(D, 1, sigma=2)

valss, vecss = eigsh(D, k=3)

#print(vals)
print()
print(valss)