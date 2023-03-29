"""
    ./code/triangular_lattice/qspace.py

    Author: Fabian R. Lux
    Date:   01/12/2023

    Determine wavevectors which are commensurate with the lattice
    structure.
"""
from fractions import Fraction
import numpy as np


def frac_to_float(frac):
    """
       Convert an instance of Fraction to a numpy float
    """

    return np.float64(frac.numerator / frac.denominator)


class QSpace():

    def __init__(self, system_sizes):
        """
           Generate a list of q vectors which are 
           compatible with the system_sizes

           systems_sizes is a 1D list of integers
        """

        system_sizes = np.sort(system_sizes)[::-1]

        qs = []
        ns = []

        for n in system_sizes:

            qn = self.__generate_qs(n)

            for q in qn:
                if q in qs:
                    pass
                else:
                    qs.append(q)
                    ns.append(n)

        self.qs = qs
        self.ns = ns
        self.nq = len(qs)

    def get_qlist(self):
        """
           Returns qs, ns as sorted numpy arrays
        """

        qfloat = [frac_to_float(q) for q in self.qs]

        zipped = np.array(list(zip(qfloat, self.ns)))
        qsorted = zipped[zipped[:, 0].argsort()]

        return np.array(qsorted[:, 0], dtype=np.float64), np.array(qsorted[:, 1], dtype=np.integer)

    def __generate_qs(self, n):
        """
           Generate q values in the open interval (0,1/2) 
           which are compatible with the system size n
        """

        if n % 2 == 0:
            max_q = n//2 - 1
        else:
            max_q = n//2

        qs = []
        for q in range(1, max_q):
            qs.append(Fraction(q, n))

        return qs
