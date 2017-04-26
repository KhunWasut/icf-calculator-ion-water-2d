from libc.stdlib cimport atof
import os


cdef inline double _str_to_double(char* s):
    return atof(s)


cdef unicode tounicode(char* s):
    return s.decode('utf-8', 'strict')


def read_split_xyz(char* path_bigfile, char* dir_prefix_splitfiles, int num_atoms, int dir_index, int num_frames_per_index):
    # SPLIT LARGE XYZ FILE AND EXTRACT ONLY COORDINATES IN .VEC FORMAT
    # C-definitions
    cdef int counter = 0
    cdef int frame_count
    cdef char* local_x_thisatom
    cdef char* local_y_thisatom
    cdef char* local_z_thisatom

    file_obj = open(tounicode(path_bigfile), 'r')

    for line in file_obj:
        # READ THE FILE IF (counter % (num_atoms+2)) >= 2 
        if (counter % (num_atoms + 2)) >= 2:
            frame_count = counter / (num_atoms + 2)
            split_line = line.split()
            if (counter % (num_atoms + 2) == 2):
                write_obj = open(os.path.join(tounicode(dir_prefix_splitfiles), 'x{0}.vec'.format(frame_count + 1 +\
                        ((dir_index-1)*num_frames_per_index))), 'w')

            local_x_this_atom = bytearray(split_line[1], 'utf-8')
            local_y_this_atom = bytearray(split_line[2], 'utf-8')
            local_z_this_atom = bytearray(split_line[3], 'utf-8')

            write_obj.write('{0:.6e}\n'.format(_str_to_double(local_x_this_atom)))
            write_obj.write('{0:.6e}\n'.format(_str_to_double(local_y_this_atom)))
            write_obj.write('{0:.6e}\n'.format(_str_to_double(local_z_this_atom)))

            if (counter % (num_atoms + 2) == (num_atoms + 2 - 1)):
                write_obj.close()

        counter = counter + 1

    file_obj.close()
