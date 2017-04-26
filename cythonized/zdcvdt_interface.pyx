from zdcvdt cimport calc_zdcvdt_oneframe, tounicode
import numpy as np
cimport numpy as np
import pandas as pd
import os


def zdcvdt_matrix_interface(int dir_index, long nframes, int stride, char* coordfiles_dir_prefix, int natoms, int cation_index, int anion_index,
        np.ndarray[np.int_t] water_indices, int nwater, double n, double m, double r0, double L, np.ndarray[np.float_t, ndim=2] mu_inv, 
        np.ndarray[np.float_t, ndim=2] dcvdt_all):

    # Now will use 5-point stencil!!
    cdef long x_prev_2h_index, x_prev_h_index, x_next_h_index, x_next_2h_index
    cdef int is_first, is_last, total_num_zdcvdt, counter
    cdef long i

    cdef np.ndarray[np.float_t] X_prev_2h = np.empty(3*natoms)
    cdef np.ndarray[np.float_t] X_prev_h = np.empty(3*natoms)
    cdef np.ndarray[np.float_t] X_next_h = np.empty(3*natoms)
    cdef np.ndarray[np.float_t] X_next_2h = np.empty(3*natoms)
    cdef np.ndarray[np.float_t] X_first_h = np.empty(3*natoms)
    cdef np.ndarray[np.float_t] X_first_2h = np.empty(3*natoms)

    cdef np.ndarray[np.float_t] zdcvdt1, zdcvdt2, zdcvdt3, zdcvdt4, zdcvdt5, zdcvdt6

    # All entries that will be recorded (including the first 2 frames)
    total_num_zdcvdt = (nframes/stride)*4

    # 3 columns - first column will store indices
    cdef np.ndarray[np.float_t, ndim=2] zdcvdt_all = np.empty((total_num_zdcvdt, 3))

    counter = 0

    # Loop within dir_index
    for i in range(nframes/stride): 
        # Calculate indices of coordinate files (1-based)
        x_prev_2h_index = nframes*(dir_index-1) + ((i+1)*stride) - 2
        x_prev_h_index = nframes*(dir_index-1) + ((i+1)*stride) - 1
        x_next_h_index = nframes*(dir_index-1) + ((i+1)*stride) + 1
        x_next_2h_index = nframes*(dir_index-1) + ((i+1)*stride) + 2
        x_first_h_index = nframes*(dir_index-1) + 1
        x_first_2h_index = nframes*(dir_index-1) + 2

        # ALL THESE X'S ARE IN ANGSTROMS BUT CALCULATIONS HAS BEEN SET UP IN ATOMIC UNITS!
        # CONVERT THESE X'S FIRST!!
        if (i != (nframes/stride)-1):
            X_prev_2h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_prev_2h_index)), header=None).as_matrix().ravel()
            X_prev_h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_prev_h_index)), header=None).as_matrix().ravel()
            X_next_h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_next_h_index)), header=None).as_matrix().ravel()
            X_next_2h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_next_2h_index)), header=None).as_matrix().ravel()
            X_prev_2h /= 0.529177
            X_prev_h /= 0.529177
            X_next_h /= 0.529177
            X_next_2h /= 0.529177
        elif (i == (nframes/stride)-1):
            X_prev_2h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_prev_2h_index)), header=None).as_matrix().ravel()
            X_prev_h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_prev_h_index)), header=None).as_matrix().ravel()
            X_prev_2h /= 0.529177
            X_prev_h /= 0.529177
        elif (i == 0):
            X_first_h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_first_h_index)), header=None).as_matrix().ravel()
            X_first_2h = pd.read_csv(os.path.join(tounicode(coordfiles_dir_prefix), 'x{0}.vec'.format(x_first_2h_index)), header=None).as_matrix().ravel()
            X_first_h /= 0.529177
            X_first_2h /= 0.529177

        # Calculate and record zdcvdt case by case
        if (i == 0):
            zdcvdt1 = np.empty(2)
            zdcvdt2 = np.empty(2)
            zdcvdt3 = np.empty(2)
            zdcvdt4 = np.empty(2)
            zdcvdt5 = np.empty(2)
            zdcvdt6 = np.empty(2)

            # If dcvdt are accumulated, we will pad the very first 2 points and very last 2 points here since we won't use them anyway
            # They can't be used with 5-point stencils!
            # index of dcvdt_all is zero-based, so subtract the calculated index with 1 to access the same element
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_first_h_index-1,:], zdcvdt1)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_first_2h_index-1,:], zdcvdt2)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_2h_index-1,:], zdcvdt3)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_h_index-1,:], zdcvdt4)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_next_h_index-1,:], zdcvdt5)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_next_2h_index-1,:], zdcvdt6)

            zdcvdt_all[counter, 0] = x_first_h_index
            zdcvdt_all[counter, 1] = zdcvdt1[0]
            zdcvdt_all[counter, 2] = zdcvdt1[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_first_2h_index
            zdcvdt_all[counter, 1] = zdcvdt2[0]
            zdcvdt_all[counter, 2] = zdcvdt2[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_prev_2h_index
            zdcvdt_all[counter, 1] = zdcvdt3[0]
            zdcvdt_all[counter, 2] = zdcvdt3[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_prev_h_index
            zdcvdt_all[counter, 1] = zdcvdt4[0]
            zdcvdt_all[counter, 2] = zdcvdt4[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_next_h_index
            zdcvdt_all[counter, 1] = zdcvdt5[0]
            zdcvdt_all[counter, 2] = zdcvdt5[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_next_2h_index
            zdcvdt_all[counter, 1] = zdcvdt6[0]
            zdcvdt_all[counter, 2] = zdcvdt6[1]
            counter += 1

        elif (i != (nframes/stride)-1):
            zdcvdt1 = np.empty(2)
            zdcvdt2 = np.empty(2)
            zdcvdt3 = np.empty(2)
            zdcvdt4 = np.empty(2)
            
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_2h_index-1,:], zdcvdt1)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_h_index-1,:], zdcvdt2)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_next_h_index-1,:], zdcvdt3)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_next_2h_index-1,:], zdcvdt4)
            
            zdcvdt_all[counter, 0] = x_prev_2h_index
            zdcvdt_all[counter, 1] = zdcvdt1[0]
            zdcvdt_all[counter, 2] = zdcvdt1[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_prev_h_index
            zdcvdt_all[counter, 1] = zdcvdt2[0]
            zdcvdt_all[counter, 2] = zdcvdt2[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_next_h_index
            zdcvdt_all[counter, 1] = zdcvdt3[0]
            zdcvdt_all[counter, 2] = zdcvdt3[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_next_2h_index
            zdcvdt_all[counter, 1] = zdcvdt4[0]
            zdcvdt_all[counter, 2] = zdcvdt4[1]
            counter += 1

        elif(i == (nframes/stride)-1):
            zdcvdt1 = np.empty(2)
            zdcvdt2 = np.empty(2)

            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_2h_index-1,:], zdcvdt1)
            calc_zdcvdt_oneframe(X_prev_2h, natoms, cation_index, anion_index, water_indices, nwater, n, m, r0, L, mu_inv.ravel(), 
                    dcvdt_all[x_prev_h_index-1,:], zdcvdt2)

            zdcvdt_all[counter, 0] = x_prev_2h_index
            zdcvdt_all[counter, 1] = zdcvdt1[0]
            zdcvdt_all[counter, 2] = zdcvdt1[1]
            counter += 1

            zdcvdt_all[counter, 0] = x_prev_h_index
            zdcvdt_all[counter, 1] = zdcvdt2[0]
            zdcvdt_all[counter, 2] = zdcvdt2[1]
            counter += 1

    return zdcvdt_all
