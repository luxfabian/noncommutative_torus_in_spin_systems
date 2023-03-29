import numpy as np
from numba import jit

from model import spectrum, construct_hilbert_space
import datetime
import sys
import os
import time
import json

from mpicore import MPIControl
from qspace import QSpace

#----------------------------------------------------------------------
# setup mpi
#----------------------------------------------------------------------

mpiv = MPIControl()
mpiv.start_clock()

#----------------------------------------------------------------------
# setup model
#----------------------------------------------------------------------

params = {
   'texture': 'skx',
   'system_sizes': [n for n in range(5,10)],
   't': -1, #hopping
   'm': 5   #exchange
}

#----------------------------------------------------------------------
# generate output outdirectory
#----------------------------------------------------------------------

outdir = ''
if mpiv.is_root():

   #unique time stamp
   time.sleep(1)
   stamp = str(round(time.time()))

   #output outdirectory
   outdir = './out/'+stamp
   os.mkdir(outdir)

   with open(outdir+'/params.json', 'w') as f:
      json.dump(params, f)

outdir = mpiv.comm.bcast(outdir, root=0)

#----------------------------------------------------------------------
# calculate the spectrum
#----------------------------------------------------------------------

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

      labels, states = construct_hilbert_space(ns[i], ns[i])

      eigvals = 2*(ns[i]*ns[i])

      spec = np.zeros(eigvals, dtype=np.float64)
      spec = spectrum(states, ns[i], ns[i], qs[i], params['t'], params['m'], params['texture'])

      key = (str(i).zfill(4))
      np.save(outdir+"/spec_"+key+".npy", spec)


#---------------------------------------------------------------------
# finalize
#----------------------------------------------------------------------

mpiv.barrier()
mpiv.print("Done!")
mpiv.stop_clock()

if mpiv.is_root():

   walltime = mpiv.get_time()
   cputime  = mpiv.size * mpiv.get_time() / 3600.0

   mpiv.print("Walltime: ", walltime)
   mpiv.print("CPUs: ", mpiv.size * mpiv.get_time())
   mpiv.print("outdir: ", outdir)
   np.savetxt(outdir+"/cputime.txt", [cputime])

mpiv.finalize()
