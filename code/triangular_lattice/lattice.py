"""
    ./code/triangular_lattice/lattice.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Geometry of the triangular lattice
"""
import numpy as np
from numba import jit
from constants import pi

# triangular lattice
a = np.array([[1.0, 0], [1/2.0, np.sqrt(3)/2.0]], dtype=np.float64)

# triangular reciprocal lattice
b = np.array([[2*pi, -2*pi / np.sqrt(3)],
             [0, 4*pi / np.sqrt(3)]], dtype=np.float64)


@jit(nopython=True)
def R(i, j):
    """
       Real-space lattice
    """

    return i*a[0] + j*a[1]


@jit(nopython=True)
def G(i, j):
    """
       Reciprocal-space lattice
    """

    return i*b[0] + j*b[1]
