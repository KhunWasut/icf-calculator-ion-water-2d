from cythonized.xyz import read_split_xyz
import numpy as np
import time, os


# DEFINITIONS
dir_index = 13
nframes_per_dir = 50000

dir_prefix_big_file = os.path.join(os.getcwd(), 'unsplit-coord-data')
big_file_filename = 'licl-const-index{0}-{1}fps-step3.xyz'.format(dir_index, nframes_per_dir)

dir_prefix_split_files = os.path.join(os.getcwd(), 'split-coord-data', 'coord-{0}-{1}steps-per-dir'.format(dir_index, nframes_per_dir))
os.mkdir(dir_prefix_split_files)

t1 = time.time()
read_split_xyz(bytearray(os.path.join(dir_prefix_big_file, big_file_filename),'utf-8'), bytearray(dir_prefix_split_files, 'utf-8'),
        230*3+2, dir_index, 50000)
t2 = time.time()
print('The time it takes to read 50,000 snapshots is {0:.4f} minutes'.format((t2-t1)/60.0))
