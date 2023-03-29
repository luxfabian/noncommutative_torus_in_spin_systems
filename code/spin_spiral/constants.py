"""
    ./code/chain/constants.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Collection of constant scalars and matrices relevant to the project.
"""
import numpy as np

# -- math constants ----------------------------------------------------

pi = np.pi

# -- pauli matrices ----------------------------------------------------

s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

# -- vector operations ----------------------------------------------------

# rotation by 120 degrees around z
rot120 = np.array([[-1/2.0, -np.sqrt(3)/2.0, 0],
                   [np.sqrt(3)/2.0, -1/2.0, 0],
                   [0, 0, 1]])

# rotation by 240 degrees around z
rot240 = np.array([[-1/2.0, +np.sqrt(3)/2.0, 0],
                   [-np.sqrt(3)/2.0, -1/2.0, 0],
                   [0, 0, 1]])