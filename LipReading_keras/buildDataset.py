from Data.Preprocessing import *


## choose your path to your directory with images, make sure to end it with "/" symbol
path = "/home/usr/lipread_mp4/"

# activating our main class, which automatically saves file
data = dataset_builder(path)

# Start to build train set
train_path = "/home/usr/train_set.hdf5"
data.get_data(name = 'train',train_path)

# Start to build train set
val_path = "/home/usr/val_set.hdf5"
data.get_data(name = 'val',val_path)

# Start to build train set
test_path = "/home/usr/test_set.hdf5"
data.get_data(name = 'test',test_path)
