import ED
from scipy.sparse.linalg import eigs, eigsh

#Example Case: Single Particle on a 2 X 2 square lattice

#Initialize Hamiltonian object with required parameters w.r.t the given order
# (M, N, T, U, lac(required only for 2D case) ) in which
# M = # of sites, N = # of particles, T = hopping strength, U = interaction strength
# lac = this parameter is only required when setting up 2D lattice in our case it is 2 since
# we are dealing with 2 X 2 square lattice amounting to 4 sites
H = ED.Hamiltonian(4, 1, 1, 1, 2)

#Forming interaction hamiltonian and hopping hamiltonian
H_int = H.H_int()
H_kin = H.H_kin(H.lattice2D()) #!!! In 1D case make sure to use lattice1D() function and leave empty the
                               # lac parameter above

H_tot = (H_int.tocsr() + H_kin.tocsr())

#One can compute the dimension of hilbert space so that the Lanczos can look for such number of eigenvalues
print(H.D()) #It will result in 4 in our case, we configured the functions below to look for 2 eigenvalues

#This method is more efficient for computing low lying eigenvalues :
eig_val,eig_vecs = eigs(H_tot, 2, sigma=-5) # Looking for 2 eigenvalues the arg < H.D() condition must be satisfied!

#More generic way w/o any specification of the spectrum
eig_val2, eig_vecs2 = eigsh(H_tot, k=2) # Looking for 2 eigenvalues: k < H.D() condition must be satisfied!


#Printing computed eigenvalues
print(ED.np.around(eig_val, decimals=3))
print(ED.np.around(eig_val2, decimals=3))

