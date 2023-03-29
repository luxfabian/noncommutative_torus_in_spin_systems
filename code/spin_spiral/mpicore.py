"""
    ./code/spin_spiral/mpicore.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Tools build around mpi4py.
"""
import time
import sys
import numpy as np

from mpi4py import MPI


class MPIControl:
    """
    Initializes the MPI communicator and implements some useful tools
    for working with MPI. This includes a clock on the root rank for 
    timing purposes and printing functions.
    """

    # -- initialize -----------------------------------------------------

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = 0
        self.err = 0

        self._t0 = 0.0
        self._t1 = 0.0
        self._total = 0.0

    # -- finalize -------------------------------------------------------

    def finalize(self):
        """
            Resolve the MPI communicator
        """
        MPI.Finalize()

    # -- barrier --------------------------------------------------------

    def barrier(self):
        """
            Blockades further code evolution until all ranks reached the
            point where the barrier is invoked.
        """
        self.comm.Barrier()

    # -- timing ---------------------------------------------------------

    def start_clock(self):
        """
            Start timing on the root rank
        """
        if self.rank == self.root:
            self._t0 = time.time()

    def stop_clock(self):
        """
            Stop timing on the root rank
        """
        if self.rank == self.root:
            self._t1 = time.time()
            self._total = self._t1 - self._t0

    def get_time(self):
        """
            Returns the timing measured by root
        """
        if self.rank == self.root:
            return self._total

    # -- printing -------------------------------------------------------

    def print(self, *args):
        """
            Printing function which only prints on root
        """
        if self.rank == self.root:
            print(*args)
            sys.stdout.flush()

    # -- rank identification --------------------------------------------

    def is_root(self):
        """
            Returns true if the current MPI rank is the designated root.
        """
        return (self.rank == self.root)

    def my_turn(self, i):
        """
            Used for uniform load balancing. Checks whether i is assigned 
            to the current rank by cycling through the integer modulo
            the number of ranks.
        """
        return (i % self.size) == self.rank

    def random_assignment(self, i):
        """
            Randomly assign i to one of the available MPI ranks.
        """
        # ISO/IEC 9899 LCG
        a = 1103515245  # multiplier
        c = 12345  # increment
        m = 2**32  # modulus

        return ((a*i+c) % m) % self.size == self.rank

    # -- communication --------------------------------------------------

    def reduce_sum(self, arr, arr_red):
        """
            Reduce the input array arr on the root by adding up the
            results of individual MPI ranks.
        """

        # extract data type
        if arr.dtype == np.dtype(np.float64):
            dtype = MPI.DOUBLE
        elif arr.dtype == np.dtype(np.int32):
            dtype = MPI.INT
        else:
            self.print("MPI ERROR: unknown dtype")
            self.err = 1
            exit(-1)

        self.comm.Reduce(
            [arr, dtype],
            [arr_red, dtype],
            op=MPI.SUM,
            root=self.root
        )

    def broadcast(self, arr):
        """
            Communicate an array on root to all MPI ranks.
        """
        return self.comm.bcast(arr, root=self.root)

    def gather(self, data):
        """
            Gather data from all ranks on root.
        """
        return self.comm.gather(data, root=self.root)

    # -- dynamic load balancing -----------------------------------------

    def assign_work(self, i):
        """
            Assign the job i to whoever is available. Available ranks
            need to listen to whoever assigns the job.
        """
        # find available worker unit
        worker_unit = self.comm.recv(source=MPI.ANY_SOURCE)
        # send some work to it
        self.comm.send(i, dest=worker_unit)

    def stop_working_units(self):
        """
            Tell working units to stop listening to root.
        """
        for i in range(self.size - 1):
            worker_unit = self.comm.recv(source=MPI.ANY_SOURCE)
            self.comm.send(-1, dest=worker_unit)


def test_mpi():
    """
        Check if ranks are live.
    """

    mpiv = MPIControl()

    print("A warm hello from: ", mpiv.rank, "/", mpiv.size)


if __name__ == '__main__':
    test_mpi()
