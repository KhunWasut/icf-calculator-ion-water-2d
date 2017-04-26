from cythonized.colvars_interface import get_colvar_mat
import numpy as np
import pandas as pd
import os
import time


# DEFINITIONS
nframes_per_dir = 50000
first_dir_index = 31
last_dir_index = 40
li_index = 0
cl_index = 1
nwater = 230
natoms = 3*nwater + 2
o_indices = np.array([3*i+2 for i in range(nwater)]).astype(np.int)

n = 20.0
m = 34.0 
L_angs = 19.033
r0_angs = 2.59

for i in range(first_dir_index, last_dir_index+1):
    coord_dir_prefix = os.path.join(os.getcwd(), 'split-coord-data', 'coord-{0}-{1}steps-per-dir'.format(i, nframes_per_dir))
    t1 = time.time()
    # The function read pure coords in generic .xyz format (angstroms), so calculate C.V. in conventional units first!
    cv = get_colvar_mat(i, nframes_per_dir, bytearray(coord_dir_prefix, 'utf-8'), li_index, cl_index, o_indices, nwater, 
            natoms, n, m, r0_angs, L_angs)
    t2 = time.time()

    print('The time it takes to compute C.V. matrix of dir_index = {0} for 50,000 frames is {1:.4f} minutes'.format(i, (t2-t1)/60.0))

    # Conversion of R to atomic unit
    cv[:,1] /= 0.529177

    cv_dir_prefix = os.path.join(os.getcwd(), 'cv-traj-data')
    cv_filename = 'cvtraj-2d-{0}-{1}-to-{2}.dat'.format(i, (nframes_per_dir*(i-1))+1, nframes_per_dir*i)
    cv_df = pd.DataFrame({'index': cv[:,0].astype(np.int), 'cvr_atomic': cv[:,1], 'cvcn': cv[:,2]})
    cv_df = cv_df[['index', 'cvr_atomic', 'cvcn']]
    cv_df.to_csv(os.path.join(cv_dir_prefix, cv_filename), index=False, header=False, sep='\t', float_format='%.10e')
