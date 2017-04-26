from colvars cimport get_cn, get_rij
import pandas as pd
import numpy as npp
import os
cimport numpy as np


cdef unicode tounicode(char* s):
    return s.decode('utf-8', 'strict')


# EXPECT ATOMIC UNITS!!
# CONVERSION DONE IN DRIVER FILE!!
def get_colvar_mat(int dir_index, long nframes_per_dir, char* coord_dir_prefix, int cation_index, int anion_index, np.ndarray[np.int_t] water_indices,
        int nwater, int natoms, double n, double m, double r0, double L):

    cdef long index, i
    cdef np.ndarray[np.float_t] X
    cdef np.ndarray[np.float_t, ndim=2] cv = npp.empty((nframes_per_dir, 3))
    cdef double [:] X_buff
    cdef double rij_thisframe, cn_thisframe
    cdef long [:] water_indices_2 = water_indices

    for i in range(nframes_per_dir):
        index = (nframes_per_dir*(dir_index-1)) + i + 1
        X = pd.read_csv(os.path.join(tounicode(coord_dir_prefix), 'x{0}.vec'.format(index)), header=None).as_matrix().ravel()
        X_buff = X

        rij_thisframe = get_rij(&X_buff[0], cation_index, anion_index, L)
        cn_thisframe = get_cn(&X_buff[0], cation_index, &water_indices_2[0], nwater, n, m, r0, L)

        cv[i, 0] = index
        cv[i, 1] = rij_thisframe
        cv[i, 2] = cn_thisframe

    return cv
