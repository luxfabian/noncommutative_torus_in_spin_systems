
"""
    ./code/chain/model.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Sets up the Hamiltonian of the model and prepares for its
    diagonalization.
"""
import numpy as np
from numba import jit

from constants import pi, sx, sy, sz


# -- hilbert space -----------------------------------------------------

@jit(nopython=True)
def construct_hilbert_space(n_x):
    """
       systematic labeling of quantum states
    """

    labels = np.zeros((n_x*2, 2), dtype=np.integer)
    states = np.zeros((n_x, 2), dtype=np.integer)

    # construct hilbert space
    ii = 0
    for i in range(n_x):
        for s in range(2):  # spin
            labels[ii] = np.array([i, s])
            states[i, s] = ii
            ii += 1

    return labels, states


# -- magnetic texture --------------------------------------------------

@jit(nopython=True)
def magnetization(i, theta_1, phi_1, theta_2, phi_2, gamma=1.0, texture='limacon'):
    """
       Unraveling of the noncommutative torus onto the skyrmion lattice
    """

    n = np.zeros(3, dtype=np.float64)

    if texture == 'limacon':
        # skyrmion triple-Q phase
        c1 = np.cos(2*pi * (i * theta_1 + phi_1))
        s1 = np.sin(2*pi * (i * theta_1 + phi_1))

        c2 = np.cos(2*pi * (i * theta_2 + phi_2))
        s2 = np.sin(2*pi * (i * theta_2 + phi_2))

        n = np.array([0, c1, s1], dtype=np.float64) + gamma * \
            np.array([0, c2, s2], dtype=np.float64)

    elif texture == 'fm':
        n = np.array([0, 0, 1], dtype=np.float64)

    return n


# -- hamiltonian setup -------------------------------------------------

@jit(nopython=True)
def nearest_neighbors(i, n_x, pbc):

    nn = []

    if pbc:
        nn.append((i+1) % n_x)
    elif (i+1) < n_x:
        nn.append((i+1))

    return np.array(nn)


@jit(nopython=True)
def set_hamiltonian(hamiltonian, states, n_x, theta_1, phi_1, theta_2, phi_2, gamma, t, m, texture, pbc):
    """
       nearest neighbor hopping plus exchange
    """

    for i in range(n_x):
        # hopping
        nn = nearest_neighbors(i, n_x, pbc)
        for n in nn:
            for s in range(2):
                ii = states[i, s]
                jj = states[n, s]

                hamiltonian[ii, jj] = t
                hamiltonian[jj, ii] = t

        # onsite term
        nvec = magnetization(i, theta_1, phi_1, theta_2, phi_2, gamma, texture)
        onsite = m * (nvec[0]*sx + nvec[1]*sy + nvec[2]*sz)
        for a in range(2):
            for b in range(2):
                ii = states[i, a]
                jj = states[i, b]
                hamiltonian[ii, jj] = onsite[a, b]


# -- linear algebra ----------------------------------------------------

def spectrum(states, n_x, theta_1, phi_1, theta_2, phi_2, gamma, t, m, texture, pbc):
    """
       calculate the spectrum by exact diagonalization
    """

    dim = 2*n_x
    H = np.zeros((dim, dim), dtype=np.complex128)
    set_hamiltonian(H, states, n_x, theta_1, phi_1, theta_2,
                    phi_2, gamma, t, m, texture, pbc)

    return np.linalg.eigvalsh(H, UPLO='U').real
