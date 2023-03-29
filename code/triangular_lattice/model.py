import numpy as np
from numba import jit

from lattice import R, G
from constants import pi,s0,sx,sy,sz,rot120,rot240


#-- hilbert space -----------------------------------------------------

@jit(nopython=True)
def construct_hilbert_space(n_x, n_y):
   """
      systematic labeling of quantum states
   """

   labels = np.zeros( (n_x*n_y*2, 3), dtype=np.integer )
   states = np.zeros( (n_x, n_y, 2), dtype=np.integer  )
   
   #construct hilbert space
   ii = 0
   for i in range(n_x):
      for j in range(n_y):
         for s in range(2): #spin
            labels[ii]    = np.array( [i,j,s] )
            states[i,j,s] = ii
            ii += 1

   return labels, states


#-- magnetic texture --------------------------------------------------

@jit(nopython=True)
def magnetization(i, j, theta, texture='skx'):
   """
      Unraveling of the noncommutative torus onto the skyrmion lattice
   """

   x = R(i,j)
   n = np.zeros(3, dtype=np.float64)

   if texture=='skx':
      #skyrmion triple-Q phase 

      phase_1 =  ( theta*np.dot(x , G(0,1) ) )   
      phase_2 =  ( theta*np.dot(x , G(-1,-1) ) ) + pi
      phase_3 =  ( theta*np.dot(x , G(1, 0) ) )

      n1 = np.array( [0, np.sin(phase_1), np.cos(phase_1)] )
      n2 = np.dot( rot120, np.array( [0, np.sin(phase_2), np.cos(phase_2)] ) )
      n3 = np.dot( rot240, np.array( [0, np.sin(phase_3), np.cos(phase_3)] ) ) 

      n =  n1 + n2 + n3
      n = n / np.linalg.norm(n)


   elif texture=='sdw':
      #spin-density wave triple-Q phase 

      phase_1 =  ( theta*np.dot(x , G(0,1) ) )   
      phase_2 =  ( theta*np.dot(x , G(-1,-1) ) ) + pi
      phase_3 =  ( theta*np.dot(x , G(1, 0) ) ) 

      n1 = np.array( [np.sin(phase_1), 0, 0 ] )
      n2 = np.dot( rot120, np.array( [np.sin(phase_2), 0, 0 ] ) )
      n3 = np.dot( rot240, np.array( [np.sin(phase_3), 0,0 ] ) ) 

      n = ( n1 + n2 + n3 )  / 3.0

   elif texture=='fm':
      n = np.array([0,0,1], dtype=np.float64)
   
   return n 


#-- hamiltonian setup -------------------------------------------------

@jit(nopython=True)
def nearest_neighbors(i, j, n_x, n_y):
   """
      Returns a list of nearest neighbors to to the site i,j. 
      The last number in each entry specify which translation was performed
   """

   nn = [  
      [ (i+1) % n_x, j],
      [ i, (j+1) % n_y],
      [ (i+1) % n_x, (j-1) % n_y]
   ] 

   return np.array(nn)

@jit(nopython=True)
def set_hamiltonian(hamiltonian, states, n_x, n_y, theta, t, m, texture):
   """
      nearest neighbor hopping on the triangular lattice plus exchange
   """

   for i in range(n_x):
      for j in range(n_y):
         
         #hopping 
         nn = nearest_neighbors(i,j,n_x,n_y)
         for site in nn:
            for s in range(2):
               ii = states[i,j,s]
               jj = states[site[0], site[1], s]

               hamiltonian[ii, jj] = t
               hamiltonian[jj, ii] = t

         #onsite term
         nvec   = magnetization(i,j,theta,texture)
         onsite = m * (nvec[0]*sx + nvec[1]*sy + nvec[2]*sz)
         for a in range(2):
            for b in range(2):
               ii = states[i,j,a]
               jj = states[i,j,b]
               hamiltonian[ii, jj] = onsite[a,b]
               hamiltonian[jj, ii] = onsite[b,a]


#-- linear algebra ----------------------------------------------------

def spectrum(states, n_x, n_y, theta, t, m, texture):
   """
      calculate the spectrum by exact diagonalization
   """
   
   dim = 2*n_x*n_y
   H = np.zeros((dim,dim), dtype=np.complex128)
   set_hamiltonian(H, states, n_x, n_y, theta, t, m, texture)

   return  np.linalg.eigvalsh(H, UPLO='U').real 