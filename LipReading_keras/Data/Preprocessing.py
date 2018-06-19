import os
import cv2
import numpy as np

def build_file_list(directory, set_name):
    labels = os.listdir(directory)

    completeList = []

    for i, label in enumerate(labels):

        dirpath = directory + "/{}/{}".format(label, set_name)

        files = os.listdir(dirpath)

        for file in files:
            if file.endswith("mp4"):
                filepath = dirpath + "/{}".format(file)
                entry = (i, filepath)
                completeList.append(entry)

    return completeList


def norm_(array, mean=0.4161, std=0.1688, with_std=False):
    '''
    Normalization of the dataset using mean and std
    if you want with std put [with_std] = True
    '''
    array = array / 255
    array -= mean
    if with_std:
        array = array / std
    return array


def video2data(filename):
    video_cap = cv2.VideoCapture(filename)
    success, image = video_cap.read()
    gray = [cv2.cvtColor(image[100:212, 80:192], cv2.COLOR_BGR2GRAY)]
    while success:
        success, image = video_cap.read()
        if success:
            temp = cv2.cvtColor(image[120:200, 80:160], cv2.COLOR_BGR2GRAY)
            gray.append(cv2.resize(temp, (112, 112)))

    frames = np.array(gray)
    frames = norm_(frames)
    return frames[:, :, :, np.newaxis]