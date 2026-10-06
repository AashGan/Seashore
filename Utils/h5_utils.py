import h5py


def read_and_rename(h5path,name_map_dict):
    """
    Function that renames h5 files 
    """
    with h5py.File(h5path,"r+") as h5file:
        for key in h5file.keys():
            for old_name,new_name in name_map_dict.items():
                if old_name in h5file[key].keys():
                    h5file[key].move(old_name,new_name)



def combine_h5(h5_1_file,h5_2_file,final_h5):
    """
    Function that combines multiple h5 files 
    """
    with h5py.File(h5_1_file, "r") as h5_1, \
         h5py.File(h5_2_file, "r") as h5_2, \
         h5py.File(final_h5, "w") as h5_out:
        for key in h5_1.keys():
            h5_1.copy(key, h5_out)

        for key in h5_2.keys():
            print(key)
            for sub_key in h5_2[key].keys():
                h5_2[key].copy(sub_key, h5_out[key])
def copy_h5(source_h5,dest_h5):
    for key in source_h5.keys():
        source_h5.copy(key,dest_h5)
