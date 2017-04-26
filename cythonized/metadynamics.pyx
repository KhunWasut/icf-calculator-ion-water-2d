from cmath cimport exp
from cython_gsl cimport *
import numpy as np
cimport numpy as np


cdef local_gaussian_exp(gsl_vector* xdiff, gsl_vector* sig):
    cdef double sum_exp = 0.0
    cdef int i

    for i in range(xdiff->size):
        sum_exp += ((gsl_vector_get(xdiff,i)*gsl_vector_get(xdiff,i))/(gsl_vector_get(sig,i)*gsl_vector_get(sig,i)))

    return exp(-0.5*sum_exp)


# WILL DESIGN THIS TO WORK BOTH WITH CONSTANT CASE AND ON-THE-FLY CASE.
# GAUSSIAN MATRICES / VECTORS TRUNCATION ARE TO BE PROCESSED IN THE INTERFACE!
cdef void get_2d_mtd_bias_namd(int natoms, int ncv, long ngauss, np.ndarray[np.float_t] W, np.ndarray[np.float_t] sig, 
        np.ndarray[np.float_t] xbar, np.ndarray[np.float_t] CV, np.ndarray[np.float_t] k_boundary, np.ndarray[np.float_t] lbounds,
        np.ndarray[np.float_t] ubounds, np.ndarray[np.float_t] mtd_bias):
    # mtd_bias is already initialized beforehand to zero from the interface layer!

    cdef gsl_vector_view this_cv, this_center, this_sig, gsl_W
    cdef gsl_matrix_view gsl_cv, gsl_xbar, gsl_sig, gsl_result_sum
    cdef gsl_vector_view local_cv_view_from_source, local_result_sum
    cdef gsl_vector *local_cv_vector, *local_mtd_bias
    cdef gsl_vector_view gsl_k_boundary, gsl_lbounds, gsl_ubounds

    cdef long i, j, k
    cdef double factor, set_val

    # LINK GSL VIEWS TO ORIGINAL MATRICES. THE MATRIX VIEWS ARE NOT TO BE ALTERED!!
    gsl_sig = gsl_matrix_view_array(&sig[0], ngauss, 2)
    gsl_xbar = gsl_matrix_view_array(&xbar[0], ngauss, 2)
    gsl_cv = gsl_matrix_view_array(&cv[0], ncv, 2)
    gsl_W = gsl_vector_view_array(&W[0], ngauss)
    gsl_result_sum = gsl_matrix_view_array(&mtd_bias[0], ncv, 2)

    gsl_k_boundary = gsl_vector_view_array(&k_boundary[0], 2)
    gsl_lbounds = gsl_vector_view_array(&lbounds[0], 2)
    gsl_ubounds = gsl_vector_view_array(&ubounds[0], 2)

    for i in range(ncv):
        # ALLOCATE A NEW LOCAL CV VECTOR!!
        local_cv_view_from_source = gsl_matrix_row(&gsl_cv.matrix, i)
        local_result_sum = gsl_matrix_row(&gsl_result_sum, i)

        for j in range(ngauss):
            # EXTRACT LOCAL GAUSSIAN VALUES FOR HEIGHT, CENTERS, AND SIGMAS
            # NOTE THAT LENGTH VALUES DIRECTLY READ FROM GAUSSIAN FILES ARE NOT CONVERTED, BUT
            # THE CONVERSIONS IS ALREADY PERFORMED IN THE INTERFACE!
            local_cv_vector = gsl_vector_alloc(2)
            local_mtd_bias = gsl_vector_calloc(2)
            gsl_vector_memcpy(local_cv_vector, &local_cv_view_from_source.vector)

            this_center = gsl_matrix_row(&gsl_xbar.matrix, j)
            this_sig = gsl_matrix_row(&gsl_sig.matrix, j)

            # SUBTRACTED CV VECTOR
            # local_cv_vector is altered!
            gsl_blas_daxpy(-1.0, this_center, local_cv_vector)

            # CALCULATION OF w exp(sum gaussian) factor
            factor = gsl_vector_get(&gsl_W.vector, j) * local_gaussian_exp(local_cv_vector, &this_sig.vector)

            # FINAL CALCULATION OF LOCAL BIAS FOR THIS GAUSSIAN
            gsl_vector_memcpy(local_mtd_bias, local_cv_vector)

            for k in range(2):
                set_val = gsl_vector_get(local_mtd_bias, k)/(gsl_vector_get(&this_sig.vector, k)*gsl_vector_get(&this_sig.vector, k))
                gsl_vector_set(local_mtd_bias, k, set_val*factor)

            # ADD THIS TO 'mtd_bias'
            # 'mtd_bias' is altered!
            gsl_blas_daxpy(1.0, local_mtd_bias, &local_result_sum.vector)

            gsl_vector_free(local_cv_vector)
            gsl_vector_free(local_mtd_bias)

        # BOUNDARY CORRECTIONS
        for k in range(2):
            if (gsl_vector_get(&local_cv_view_from_source.vector, k) > gsl_vector_get(&ubounds.vector, k)):
                set_val = gsl_vector_get(&local_result_sum.vector, k) + ((-1.0)*gsl_vector_get(k_boundary, k)*
                        (gsl_vector_get(&local_cv_view_from_source.vector, k) - gsl_vector_get(&ubounds.vector, k)))
                gsl_vector_set(&local_result_sum.vector, k, set_val)
            elif (gsl_vector_get(&local_cv_view_from_source.vector, k) < gsl_vector_get(&lbounds.vector, k)):
                set_val = gsl_vector_get(&local_result_sum.vector, k) + ((-1.0)*gsl_vector_get(k_boundary, k)*
                        (gsl_vector_get(&local_cv_view_from_source.vector, k) - gsl_vector_get(&lbounds.vector, k)))
                gsl_vector_set(&local_result_sum.vector, k, set_val)
