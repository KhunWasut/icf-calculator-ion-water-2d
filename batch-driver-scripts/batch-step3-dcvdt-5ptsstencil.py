import pandas as pd
import numpy as np
import os


# DEFINITIONS
dir_index = 13
nframes_per_dir = 50000
dt = 1.0*0.75*1.0e-15/2.418884e-17

# Open cv file
cv_dir_prefix = os.path.join(os.getcwd(), 'cv-traj-data')
cv_filename = 'cvtraj-2d-{0}-{1}-to-{2}.dat'.format(dir_index, nframes_per_dir*(dir_index-1)+1, nframes_per_dir*dir_index)
if (dir_index != 1):
    cv_prev_dir_filename = 'cvtraj-2d-{0}-{1}-to-{2}.dat'.format(dir_index-1, nframes_per_dir*(dir_index-2)+1, nframes_per_dir*(dir_index-1))
cv_next_dir_filename = 'cvtraj-2d-{0}-{1}-to-{2}.dat'.format(dir_index+1, nframes_per_dir*(dir_index)+1, nframes_per_dir*(dir_index+1))
cv_mat_thisdir = pd.read_csv(os.path.join(cv_dir_prefix, cv_filename), header=None, sep='\s+').as_matrix()[:,1:3]

# Search for cv traj file from previous index and next index

if (dir_index != 1):
    if not os.path.isfile(os.path.join(cv_dir_prefix, cv_prev_dir_filename)):
        prev_exists = False
    else:
        prev_exists = True

if (dir_index == 1):
    prev_exists = False

if not os.path.isfile(os.path.join(cv_dir_prefix, cv_next_dir_filename)):
    next_exists = False
else:
    next_exists = True

# Calculate dcvdt with finite difference (5-point stencil)
# f'(x) \approx (1/12h)(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h))

dcvdt = np.empty((nframes_per_dir, 3))

for i in range(nframes_per_dir):
    # DO SOMETHING
    curr_index = nframes_per_dir*(dir_index-1) + i + 1

    # If i > 1 and i < nframes_per_dir-2, do this regularly
    if (i > 1) and (i < (nframes_per_dir-2)):
        dcvdt[i,0] = curr_index
        dcvdt[i,1] = (-cv_mat_thisdir[i+2,0] + 8*cv_mat_thisdir[i+1,0] - 8*cv_mat_thisdir[i-1,0] + cv_mat_thisdir[i-2,0])/(12*dt) 
        dcvdt[i,2] = (-cv_mat_thisdir[i+2,1] + 8*cv_mat_thisdir[i+1,1] - 8*cv_mat_thisdir[i-1,1] + cv_mat_thisdir[i-2,1])/(12*dt)
    elif (i <= 1):
        if not prev_exists:
            # PAD THE DATA IF PREV DOES NOT EXIST!
            dcvdt[i,0] = curr_index
            dcvdt[i,1] = 0.0
            dcvdt[i,2] = 0.0
        elif prev_exists:
            dcvdt[i,0] = curr_index
            cv_mat_prevdir = pd.read_csv(os.path.join(cv_dir_prefix, cv_prev_dir_filename), header=None, sep='\s+').as_matrix()[:,1:3]
            if ((curr_index - 2) < nframes_per_dir*(dir_index-1) + 1):
                if ((curr_index - 1) < nframes_per_dir*(dir_index-1) + 1):
                    # MOST EXTREME CASE WHEN 2 BACKWARD FRAMES ARE IN PREVIOUS INDEX FILE
                    dcvdt[i,1] = (-cv_mat_thisdir[i+2,0] + 8*cv_mat_thisdir[i+1,0] - 8*cv_mat_prevdir[nframes_per_dir-1,0] +
                            cv_mat_prevdir[nframes_per_dir-2,0])/(12*dt)
                    dcvdt[i,2] = (-cv_mat_thisdir[i+2,1] + 8*cv_mat_thisdir[i+1,1] - 8*cv_mat_prevdir[nframes_per_dir-1,1] +
                            cv_mat_prevdir[nframes_per_dir-2,1])/(12*dt)
                else:
                    dcvdt[i,1] = (-cv_mat_thisdir[i+2,0] + 8*cv_mat_thisdir[i+1,0] - 8*cv_mat_thisdir[0,0] + 
                            cv_mat_prevdir[nframes_per_dir-1,0])/(12*dt)
                    dcvdt[i,2] = (-cv_mat_thisdir[i+2,1] + 8*cv_mat_thisdir[i+1,1] - 8*cv_mat_thisdir[0,1] + 
                            cv_mat_prevdir[nframes_per_dir-1,1])/(12*dt)

    elif (i >= (nframes_per_dir-2)):
        if not next_exists:
            # PAD THE DATA IF NEXT DOES NOT EXIST!
            dcvdt[i,0] = curr_index
            dcvdt[i,1] = 0.0
            dcvdt[i,2] = 0.0
        elif next_exists:
            dcvdt[i,0] = curr_index
            cv_mat_nextdir = pd.read_csv(os.path.join(cv_dir_prefix, cv_next_dir_filename), header=None, sep='\s+').as_matrix()[:,1:3]
            if ((curr_index + 2) > nframes_per_dir*dir_index):
                if ((curr_index + 1) > nframes_per_dir*dir_index):
                    # MOST EXTREME CASE WHEN 2 FORWARD FRAMES ARE IN NEXT INDEX FILE
                    dcvdt[i,1] = (-cv_mat_nextdir[1,0] + 8*cv_mat_nextdir[0,0] - 8*cv_mat_thisdir[i-1,0] +
                            cv_mat_thisdir[i-2,0])/(12*dt)
                    dcvdt[i,2] = (-cv_mat_nextdir[1,1] + 8*cv_mat_nextdir[0,1] - 8*cv_mat_thisdir[i-1,1] +
                            cv_mat_thisdir[i-2,1])/(12*dt)
                else:
                    dcvdt[i,1] = (-cv_mat_nextdir[0,0] + 8*cv_mat_thisdir[nframes_per_dir-1,0] - 8*cv_mat_thisdir[i-1,0] + 
                            cv_mat_thisdir[i-2,0])/(12*dt)
                    dcvdt[i,2] = (-cv_mat_nextdir[0,1] + 8*cv_mat_thisdir[nframes_per_dir-1,1] - 8*cv_mat_thisdir[i-1,1] + 
                            cv_mat_thisdir[i-2,1])/(12*dt)


# SAVE DATA
dcvdt_df = pd.DataFrame({'index': dcvdt[:,0].astype(np.int), 'dcvdt_r': dcvdt[:,1], 'dcvdt_cn': dcvdt[:,2]})
dcvdt_df = dcvdt_df[['index', 'dcvdt_r', 'dcvdt_cn']]

dcvdt_dir_prefix = os.path.join(os.getcwd(), 'dcvdt-data') 
dcvdt_filename = 'dcvdt-2d-{0}-{1}-to-{2}.dat'.format(dir_index, nframes_per_dir*(dir_index-1)+1, nframes_per_dir*dir_index)
dcvdt_df.to_csv(os.path.join(dcvdt_dir_prefix, dcvdt_filename), sep='\t', index=False, header=False, float_format='%.10e')
