cimport numpy as np


cdef unicode tounicode(char*)
cdef void colvars_jacobian_2d_ion_water(np.ndarray[np.float64_t, ndim=1], np.ndarray[np.float64_t, ndim=1], int, int, int, 
        np.ndarray[np.int_t], int, double, double, double, double)
cdef void calc_Z(np.ndarray[np.float_t], int, int, np.ndarray[np.float_t], int, np.ndarray[np.float_t])
cdef void solve(np.ndarray[np.float_t], np.ndarray[np.float_t], np.ndarray[np.float_t], int)
cdef void calc_zdcvdt_oneframe(np.ndarray[np.float_t], int, int, int, np.ndarray[np.int_t], int, double, double, double, double, 
        np.ndarray[np.float_t], np.ndarray[np.float_t], np.ndarray[np.float_t])

