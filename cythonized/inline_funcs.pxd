cdef inline int kth_axis(int v_k_index):
    return v_k_index % 3


cdef inline int kth_atom_index(int v_k_index):
    return v_k_index / 3


cdef inline int get_kth_index(int atom_index, int axis):
    return 3*atom_index + axis
