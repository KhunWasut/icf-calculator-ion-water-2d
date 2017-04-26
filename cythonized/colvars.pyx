# C Declarations of collective variable calculations to be used in Cython
from cmath cimport sqrt, pow 
from inline_funcs cimport kth_axis, kth_atom_index, get_kth_index


cdef double get_rij(double *X, int i_index, int j_index, double L):
    # Use math.h function that has a header in cmath.pxd
    cdef double dx, dy, dz

    dx = X[get_kth_index(j_index, 0)] - X[get_kth_index(i_index, 0)]
    dy = X[get_kth_index(j_index, 1)] - X[get_kth_index(i_index, 1)]
    dz = X[get_kth_index(j_index, 2)] - X[get_kth_index(i_index, 2)]

    if (dx >= (L/2.0)):
        dx = dx - L
    elif (dx <= (-L/2.0)):
        dx = dx + L

    if (dy >= (L/2.0)):
        dy = dy - L
    elif (dy <= (-L/2.0)):
        dy = dy + L

    if (dz >= (L/2.0)):
        dz = dz - L
    elif (dz <= (-L/2.0)):
        dz = dz + L

    return sqrt(dx*dx + dy*dy + dz*dz)


cdef double get_cn(double* X, int i_index, long* j_indices, int size_j, double n, double m, double r0, double L):
    cdef double cn_sum = 0.0
    cdef int j_index
    cdef double N, D

    for j_index in range(size_j):
        N = 1.0 - pow((get_rij(X, i_index, j_indices[j_index], L)/r0), n)
        D = 1.0 - pow((get_rij(X, i_index, j_indices[j_index], L)/r0), m)
        cn_sum = cn_sum + (N/D)

    return cn_sum


cdef double calc_dr_dxk(double* X, int i_index, int j_index, int v_k_index, double L):
    # X and L should have same unit
    cdef double sign
    cdef double dX_k
    cdef int axis

    if (kth_atom_index(v_k_index) == j_index):
        sign = 1.0
    elif (kth_atom_index(v_k_index) == i_index):
        sign = -1.0
    else:
        sign = 0.0

    axis = kth_axis(v_k_index)
    dX_k = X[get_kth_index(j_index, axis)] - X[get_kth_index(i_index, axis)]

    if (dX_k > L/2.0):
        dX_k = dX_k - L
    elif (dX_k < -L/2.0):
        dX_k = dX_k + L

    return sign*dX_k/get_rij(X, i_index, j_index, L)


cdef double calc_dcn_dxk(double* X, int i_index, long* j_indices, int size_j, int v_k_index, double n, double m, double r0, double L):
    cdef double dcn_sum = 0.0
    cdef double N, D, dN, dD
    cdef double sign
    cdef double r_ij

    cdef int is_sign_not_zero
    cdef int j_index

    for j_index in range(size_j):
        if (kth_atom_index(v_k_index) == i_index):
            sign = -1.0
            is_sign_not_zero = 1
        elif (kth_atom_index(v_k_index) == j_indices[j_index]):
            sign = 1.0
            is_sign_not_zero = 1
        else:
            sign = 0.0
            is_sign_not_zero = 0

        if is_sign_not_zero:
            r_ij = get_rij(X, i_index, j_indices[j_index], L)
            N = 1.0 - pow((r_ij/r0), n)
            D = 1.0 - pow((r_ij/r0), m)
            dN = -n*(1.0 - N)*calc_dr_dxk(X, i_index, j_indices[j_index], v_k_index, L)/r_ij
            dD = -m*(1.0 - D)*calc_dr_dxk(X, i_index, j_indices[j_index], v_k_index, L)/r_ij

            dcn_sum = dcn_sum + ((D*dN - N*dD)/(pow(D, 2.0)))

    return dcn_sum
