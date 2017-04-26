from cythonized.zdcvdt_interface import zdcvdt_matrix_interface
import numpy as np
import pandas as pd
import os, time


#def zdcvdt_matrix_interface(int dir_index, long nframes, int stride, char* coordfiles_dir_prefix, int natoms, int cation_index, int anion_index,
#        np.ndarray[np.int_t] water_indices, int nwater, double n, double m, double r0, double L, np.ndarray[np.float_t, ndim=2] mu_inv, 
#        np.ndarray[np.float_t, ndim=2] dcvdt_all):
#    pass

# CALCULATE ZDCVDT
# AT THIS POINT WE NEED THE COORDINATE VECTOR FILES TO COMPUTE JACOBIANS
# AFTER THIS STEP WE CAN SAFELY DELETE (HAS TO BE TESTED FIRST) THOSE COORDINATE VECTOR FILES TO SAVE SPACE FOR CORRESPONDING DIR INDEX

# FIRST, GET THE INVERTED MASS MATRIX FROM THE INVERTED MASS DATA FILE (ALREADY IN A.U.)
main_dir_prefix = os.getcwd()
path_to_mass_file = os.path.join(main_dir_prefix, 'mass-data', 'masses.vec')
inv_mass_matrix = np.diag(pd.read_csv(path_to_mass_file, header=None).as_matrix().ravel())

# CALCULATION DEFINITION
first_index = 30
last_index = 39
fps = 50000
stride = 50
nwater = 230
natoms = 3*nwater+2
li_index = 0
cl_index = 1
o_indices = np.array([3*i+2 for i in range(nwater)]).astype(np.int)
n = 20.0
m = 34.0
r0_au = 2.59/0.529177
L_au = 19.033/0.529177

# THEN READ ALL DCVDT FILES (MAKE SURE TO READ FROM THE FIRST ONE!!)
# NAME FORMAT: 'dcvdt-2d-1-1-to-50000.dat'

for i in range(first_index, last_index+1):
    dcvdtr_all = np.array([], dtype=np.float)
    dcvdtcn_all = np.array([], dtype=np.float)
    for j in range(i):
        path_to_dcvdt_file = os.path.join(main_dir_prefix, 'dcvdt-data', 'dcvdt-2d-{0}-{1}-to-{2}.dat'.format(j+1, fps*j+1, fps*(j+1)))
        dcvdt_mat_thisindex = pd.read_csv(path_to_dcvdt_file, header=None, sep='\s+').as_matrix()[:,1:3]
        dcvdtr_thisframe = dcvdt_mat_thisindex[:,0]
        dcvdtcn_thisframe = dcvdt_mat_thisindex[:,1]
        dcvdtr_all = np.concatenate((dcvdtr_all, dcvdtr_thisframe))
        dcvdtcn_all = np.concatenate((dcvdtcn_all, dcvdtcn_thisframe))

    dcvdtr_all = dcvdtr_all.reshape(fps*i, 1)
    dcvdtcn_all = dcvdtcn_all.reshape(fps*i, 1)
    dcvdt_all = np.concatenate((dcvdtr_all, dcvdtcn_all), axis=1)

    # THE FUNCTION IS WRITTEN FOR LARGE STRIDE (RECOMMENDED AT LEAST 10)
    # GET ALL STENCIL POINTS NEEDED FOR 5-POINT STENCIL OF ZDCVDT IN NEXT STEP
    coordfile_dir_base = os.path.join(os.getcwd(), 'split-coord-data')
    coordfile_dir_name = 'coord-{0}-{1}steps-per-dir'.format(i, fps)
    coordfile_dir = os.path.join(coordfile_dir_base, coordfile_dir_name)

    t1 = time.time()
    zdcvdt_thisindex = zdcvdt_matrix_interface(i, fps, stride, bytearray(coordfile_dir, 'utf-8'), natoms, li_index, cl_index, o_indices, 
            nwater, n, m, r0_au, L_au, inv_mass_matrix, dcvdt_all)
    t2 = time.time()
    print('At dir_index = {0}, the process takes {1:.4f} minutes'.format(i, (t2-t1)/60.0))
    # OUTPUT DATA TO FILE
    zdcvdt_dir = os.path.join(os.getcwd(), 'zdcvdt-data')
    zdcvdt_fname = 'zdcvdt-2d-{0}-{1}-to-{2}.dat'.format(i, fps*(i-1)+1, fps*i)
    zdcvdt_df = pd.DataFrame({'indices': zdcvdt_thisindex[:,0].astype(np.int), 'zdcvdtr': zdcvdt_thisindex[:,1], 'zdcvdtcn': zdcvdt_thisindex[:,2]})
    zdcvdt_df = zdcvdt_df[['indices', 'zdcvdtr', 'zdcvdtcn']]
    zdcvdt_df.to_csv(os.path.join(zdcvdt_dir, zdcvdt_fname), sep='\t', header=False, index=False, float_format='%.10e')
