import os
import dlib
from imutils import face_utils
import cv2
import numpy as np
import glob
import h5py
from collections import OrderedDict

### You should download shape_predictor_68_face_landmarks.dat
### from http://dlib.net/files/

def get_labels(directory = 'lipread_mp4'):
    ''' 
    From specific file path (directory)
    collecting all label names (folder names)

    Return dictionary with numerical value for each label
    Will be used for y_train, y_val, y_test
    '''
    labels = {}
    idx = 0

    for name in os.listdir(directory):
        labels[name] = idx
        idx += 1

    labels = list(OrderedDict(sorted(labels.items(), key=lambda t: t[1])).items())
    return labels

def norm_(array, mean = 0.4161, std = 0.1688,with_std = False):
    ''' 
    Normalization of the dataset using mean and std
    if you want with std put [with_std] = True
    '''
    array = array/255
    array -= mean
    if with_std:
        array = array/std
    return array

def hdf5_saver(x_array, y_array, hdf5_path, x_name, y_name):
    '''
    Save you preprocessed array for further use
    hdf5_path example: '/home/Q/tr50k.hdf5'
    x_name  example  : 'x_train'
    y_name  example  : 'y_train'
    '''
    hdf5_file = h5py.File(hdf5_path, mode='w')
    N = x_array.shape[0]

    hdf5_file.create_dataset(x_name, (N,29,112,112,1), np.uint8)
    hdf5_file[x_name][...] = x_array

    hdf5_file.create_dataset(y_name, (N,29,1) , np.uint16)
    hdf5_file[y_name][...] = y_array
    hdf5_file.close()
    return print("File succesfully created and saved in %s" % (hdf5_path))

#############################################################################################################################
##################################################### MAIN ENGINE ###########################################################
#############################################################################################################################

def video2data(filename, face_detector, landmarks_predictor, radius=56):

    video_cap = cv2.VideoCapture(filename)
    success, image = video_cap.read()
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
    while success:
        success, image = video_cap.read()
        # print('Read a new frame: ', success)
        if success:
            gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    frames = np.array(gray)

    lenf = len(frames)
    lm4video = np.zeros((lenf, 68, 2))
    for i in range(len(frames)):
        img = frames[i]
        face = face_detector(img, 0)

        if not face:
            center = 128
            mouth = frames[:, center - radius: center + radius, center - radius: center + radius]
            return mouth

        preds = landmarks_predictor(img, face[0])
        preds = face_utils.shape_to_np(preds)
        lm4video[i, :, :] = preds

    ##### 3 steps
    # 1: find mean coordinates of all mouth landmarks
    # 2: choose median as center
    # 3: take an area of 112 x 112 pixels around center
    mouth = lm4video[:, 48:, :]
    mc = np.mean(mouth, axis=1)
    center = np.median(mc, axis=0)
    y_center = int(center[1])
    x_center = int(center[0])

    frames = np.array(frames)
    frames = norm_(frames)
    mouth = frames[:, y_center - radius: y_center + radius, x_center - radius: x_center + radius]

    return mouth


##############################################################################################################################


class dataset_builder():

    def __init__(self, path):
        self.labels = get_labels(path)
        self.path = path

        # init of pretrained model
        self.detector_dlib = dlib.get_frontal_face_detector()
        self.predictor_dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def get_data(self,name,hdf5_path):

        if(name == 'train'):
            N = 488766
        else:
            N = 25000

        self.x= np.zeros((N,29,112,112), dtype = np.uint8)
        self.y = np.zeros((N,29,1), dtype = np.uint16)
        i_0 = 0
        for label, value in self.labels:
            p4g = self.path + "/" + label + name + "*.mp4"
            video_files = glob.glob(p4g)
            video_files = video_files
            i_1 = len(video_files) + i_0
            self.y[i_0:i_1] = value
            for ix, video in enumerate(video_files):
                result = video2data(video, self.detector_dlib, self.predictor_dlib, False)
                index = i_0 + ix
                if type(result) == np.ndarray:
                    self.x[index] = result

        hdf5_saver(self.x, self.y,hdf5_path, 'x_'+name, 'y_'+name)
