# Calculation of Zdcvdt

from colvars cimport *
from cython_gsl cimport *

import numpy as np
cimport numpy as np


cdef unicode tounicode(char* s):
    return s.decode('utf-8', 'strict')

# This method interfaces with python driver code
# Auxillary methods are implemented here in C
# For ion-water systems only!
cdef void colvars_jacobian_2d_ion_water(np.ndarray[np.float64_t, ndim=1] X, np.ndarray[np.float64_t, ndim=1] J,
        int natoms, int cation_index, int anion_index, np.ndarray[np.int_t] water_indices, int nwater, double n, 
        double m, double r0, double L):

    cdef int v_k_index
    cdef long [:] j_indices = water_indices

    # Use the scheme: first column = R, second column = CN
    for v_k_index in range(3*natoms):
        J[2*v_k_index] = calc_dr_dxk(&X[0], cation_index, anion_index, v_k_index, L)
        J[2*v_k_index+1] = calc_dcn_dxk(&X[0], cation_index, &j_indices[0], nwater, v_k_index, n, m, r0, L)


cdef void calc_Z(np.ndarray[np.float_t] J, int J_nrows, int J_ncols, np.ndarray[np.float_t] mu_inv, int mu_inv_nrowscols, np.ndarray[np.float_t] Z):
    cdef gsl_matrix *tmp
    cdef gsl_matrix_view J_view, mu_inv_view, Z_view

    tmp = gsl_matrix_calloc(mu_inv_nrowscols, J_ncols)
    J_view = gsl_matrix_view_array(&J[0], J_nrows, J_ncols)
    mu_inv_view = gsl_matrix_view_array(&mu_inv[0], mu_inv_nrowscols, mu_inv_nrowscols)
    Z_view = gsl_matrix_view_array(&Z[0], J_ncols, J_ncols)

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &mu_inv_view.matrix, &J_view.matrix, 0.0, tmp)
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &J_view.matrix, tmp, 0.0, &Z_view.matrix)

    gsl_matrix_free(tmp)


cdef void solve(np.ndarray[np.float_t] Z, np.ndarray[np.float_t] dcvdt, np.ndarray[np.float_t] zdcvdt, int ncolvars):
    cdef gsl_vector_view dcvdt_view, zdcvdt_view
    cdef gsl_matrix_view Z_view
    cdef gsl_permutation *P
    cdef int s

    dcvdt_view = gsl_vector_view_array(&dcvdt[0], ncolvars)
    zdcvdt_view = gsl_vector_view_array(&zdcvdt[0], ncolvars)
    Z_view = gsl_matrix_view_array(&Z[0], ncolvars, ncolvars)

    P = gsl_permutation_alloc(ncolvars)

    # Use LU factorization to solve this for Z_inv dcvdt = zdcvdt to bypass inverse calculation
    gsl_linalg_LU_decomp(&Z_view.matrix, P, &s)
    gsl_linalg_LU_solve(&Z_view.matrix, P, &dcvdt_view.vector, &zdcvdt_view.vector)

    gsl_permutation_free(P)


# Use 'def' for any interfaces with python driver code
cdef void calc_zdcvdt_oneframe(np.ndarray[np.float_t] X, int natoms, int cation_index, int anion_index, np.ndarray[np.int_t] water_indices,
        int nwater, double n, double m, double r0, double L, np.ndarray[np.float_t] mu_inv, np.ndarray[np.float_t] dcvdt_one, np.ndarray[np.float_t] zdcvdt):

    # Create an empty matrix J and type it
    cdef np.ndarray[np.float_t] J = np.empty(3*natoms*2)
    cdef np.ndarray[np.float_t] Z = np.zeros(2*2)
    colvars_jacobian_2d_ion_water(X, J, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L)
    calc_Z(J, 3*natoms, 2, mu_inv, 3*natoms, Z)
    solve(Z, dcvdt_one, zdcvdt, 2)



