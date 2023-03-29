"""
    ./code/spin_spiral/run.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Run the simulation. Diagonalizes the Hamiltonian for a 1D spin chain
    for different wave vectors.
"""
import sys
import os
import time
import json

import numpy as np

from mpicore import MPIControl
from qspace import QSpace
from model import spectrum, construct_hilbert_space

# ----------------------------------------------------------------------
# setup mpi
# ----------------------------------------------------------------------

mpiv = MPIControl()

# ----------------------------------------------------------------------
# setup model
# ----------------------------------------------------------------------

params = {
    'texture': 'limacon',
    'system_sizes': [1024],
    't': -1,  # hopping
    'm': 1  # exchange
}

# ----------------------------------------------------------------------
# generate output outdirectory
# ----------------------------------------------------------------------

outdir = ''
if mpiv.is_root():

    # unique time stamp
    time.sleep(1)
    stamp = str(round(time.time()))

    # output outdirectory
    outdir = './out/'+stamp
    os.mkdir(outdir)

    with open(outdir+'/params.json', 'w') as f:
        json.dump(params, f)

outdir = mpiv.comm.bcast(outdir, root=0)

# ----------------------------------------------------------------------
# calculate the spectrum
# ----------------------------------------------------------------------

Q = QSpace(params['system_sizes'])

qs, ns = Q.get_qlist()
n_q = len(qs)

if mpiv.is_root():

    np.save(outdir+"/qs.npy", qs)
    np.save(outdir+"/ns.npy", qs)

for i in range(n_q):
    if mpiv.my_turn(i):
        print("current index:", i+1, n_q)
        sys.stdout.flush()

        labels, states = construct_hilbert_space(ns[i])

        eigvals = 2*(ns[i])

        spec = np.zeros(eigvals, dtype=np.float64)

        gamma = 1

        theta_1, phi_1, theta_2, phi_2 = qs[i], 0.0, 2*qs[i], 0.0
        spec = spectrum(states, ns[i], theta_1, phi_1, theta_2, phi_2,
                        gamma, params['t'], params['m'], params['texture'], pbc=True)

        key = (str(i).zfill(4))
        np.save(outdir+"/spec_"+key+".npy", spec)


# ---------------------------------------------------------------------
# finalize
# ----------------------------------------------------------------------

mpiv.barrier()
mpiv.print("Done!")
mpiv.finalize()
