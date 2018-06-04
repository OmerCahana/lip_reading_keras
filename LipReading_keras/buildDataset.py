from Data.Preprocessing import *


## choose your path to your directory with images, make sure to end it with "/" symbol
path = "/home/usr/lipread_mp4/"

# activating our main class, which automatically saves file
data = dataset_builder(path)

# Start to build train set
hdf5_path_val = "/home/usr/train_set.hdf5"
data.get_train(1000, hdf5_path_val)

# Start to build train set
hdf5_path_val = "/home/usr/val_set.hdf5"
data.get_val(50, hdf5_path_val)

# Start to build train set
hdf5_path_val = "/home/usr/test_set.hdf5"
data.get_test(50, hdf5_path_val)
