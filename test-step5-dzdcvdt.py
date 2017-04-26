# Calculate dzdcvdt from zdcvdt data (batch)
import numpy as np
import pandas as pd
import os, time


# SYSTEM INFORMATION
first_dir_index = 1
last_dir_index = 39
fps = 50000
stride = 50
dt = (0.75*1.0e-15)/(2.418884e-17)

# READ DATA AND COMBINE INTO ONE BIG ARRAY
# ZDCVDT DATA IS ALREADY STENCIL-IZED, READY FOR FINITE DIFFERENCE CALCULATION

steps_info = np.array([], dtype=np.int)
zdcvdt_r = np.array([], dtype=np.float)
zdcvdt_cn = np.array([], dtype=np.float)

for i in range(first_dir_index, last_dir_index+1):
    # SET UP PATHS TO READ DATA
    zdcvdt_dir_prefix = os.path.join(os.getcwd(), 'zdcvdt-data')
    zdcvdt_filename = 'zdcvdt-2d-{0}-{1}-to-{2}.dat'.format(i, (i-1)*fps+1, i*fps)
    # READ ALL COLUMNS!! WILL MAKE USE OF INDEX OF FIRST COLUMN!!
    data = pd.read_csv(os.path.join(zdcvdt_dir_prefix, zdcvdt_filename), sep='\s+', header=None).as_matrix()
    steps_info = np.concatenate((steps_info, data[:,0]))
    zdcvdt_r = np.concatenate((zdcvdt_r, data[:,1]))
    zdcvdt_cn = np.concatenate((zdcvdt_cn, data[:,2]))

# ACTUALLY CALCULATING FINITE DIFFERENCE DERIVATIVE
result = []

# LOOP OVER INDICES COLUMN
for i in range(steps_info.shape[0]):
    # CHECK IF i != 0 and the modulo of i and stride is 1 (pick stride*n+1 index to calculate)
    if (i != 0) and (steps_info[i] % stride == 1):
        # CALCULATION f'(x) \approx (1/12h)(-f(x+2h) + 8(x+h) - 8(x-h) + f(x-2h))
        dzdcvr_dt = 
