import os
import dlib
from imutils.video import FileVideoStream
from imutils import face_utils
import matplotlib.pyplot as plt
from skimage import io
import cv2
from PIL import Image, ImageOps
import numpy as np
import glob
from scipy import ndimage
import random
import math
from sklearn.utils import shuffle
import h5py
import time
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
    temp = {}
    idx = 0
    
    for name in os.listdir(directory):
        labels[name] = idx
        idx += 1
    
    labels = list(OrderedDict(sorted(labels.items(), key=lambda t: t[1])).items())
    return labels 




def AT_process(image, angle):
    ''' 
    Most of the function are from Augmentator library: 
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    class RotateRange which is based upod PIL.Image

    Returns rotated 20 degrees image without black area for AUGMENTATION purposes
    '''
    image = Image.fromarray(image)
    x = image.size[0]
    y = image.size[1]

    # Rotate, while expanding the canvas size
    image = image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Get size after rotation, which includes the empty space
    X = image.size[0]
    Y = image.size[1]

    # Get our two angles needed for the calculation of the largest area
    angle_a = abs(angle)
    angle_b = 90 - angle_a

    # Python deals in radians so get our radians
    angle_a_rad = math.radians(angle_a)
    angle_b_rad = math.radians(angle_b)

    # Calculate the sins
    angle_a_sin = math.sin(angle_a_rad)
    angle_b_sin = math.sin(angle_b_rad)

    # Find the maximum area of the rectangle that could be cropped
    E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
        (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
    E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
    B = X - E
    A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

    # Crop this area from the rotated image
    # image = image.crop((E, A, X - E, Y - A)), changed some values 
    image = image.crop((int(round(E)), int(round(A)-5), int(round(X - E+10)), int(round(Y - A +10))))

    # Return the image, re-sized to the size of the image passed originally
    image = image.resize((x, y), resample=Image.BICUBIC)
    return np.array(image)




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




def shuffle_(array_x, array_y):
    ''' 
    Symmetrically shuffle 2 arrays using random lib and sklearn.utils
    '''
    N = array_x.shape[0]
    state = np.random.randint(1,N)
    
    shuffle(array_x, random_state = state)
    shuffle(array_y, random_state = state)
    return (array_x, array_y)




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

    hdf5_file.create_dataset(y_name, (N,50) , np.uint16)
    hdf5_file[y_name][...] = y_array
    hdf5_file.close()
    return print("File succesfully created and saved in %s" % (hdf5_path))

#############################################################################################################################
##################################################### MAIN ENGINE ###########################################################
#############################################################################################################################

def video2data(video, face_detector, landmarks_predictor, AT = False):
    ''' 
    Our core detector, which uses 2D FACE ALIGNMENT model based on dlib
    https://github.com/1adrianb/face-alignment by Bulat, Adrian and Tzimiropoulos, Georgios

    sleep_time: usually it takes from 0.3 to 2 sec to get frames

    AT:   augmentation of transposed images(-20 or 20)

    Returns: preprocessed video in form of [29 x 112 x 112]
    of cropped mouth regions from each frame
    as well as calculates total time taken
    '''    
    start = time.time()
    # GETTING 29 FRAMES using IMUTILS
    vs = FileVideoStream(video).start()
    time.sleep(0.1)
    frames = []
    
    while vs.more():
        frame = vs.read()
        frames.append(frame)

    #if AT:
       # angle = random.choice((20,-20))
    # DETECTING FACES AND CROPPING MOUTH REGION USING 2D FACE ALIGNMENT algorithm   
    # empty container for further use (for landmarks)
    lm4video= np.zeros((29,68,2)) 

    for i in range(len(frames)):
        img = frames[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_detector(gray,0)
        if not face:
            return None
        preds = landmarks_predictor(gray,face[0])
        preds = face_utils.shape_to_np(preds)
        lm4video[i,:,:] = preds
            
    # median coordinates for all 29 video frames in order to get best crop area for all frames
    # ideal crop: 1/4 of nose, 1/2 of chin, width depends on its size
    mc = np.median(lm4video,axis=0)
    y0,y1 = int(np.mean((mc[29][1],mc[30][1]))), int(mc[7][1])
    center = int(np.mean((mc[52][0], mc[63][0], mc[67][0], mc[58][0])))
    width_from_center = (y1 - y0)//2
    x0 = center - width_from_center
    x1 = center + width_from_center
    
    idx = 0
    f2x = np.zeros((29,112,112))
    for frame in frames:
        mouth = frame[y0:y1, x0:x1]
        mouth = cv2.resize(mouth,(112,112)) 
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        #  Augmentation, rotated images
        #if AT:
            #mouth = AT_process(mouth,angle)
        f2x[idx,:,:] = mouth
        idx += 1
    return f2x

##############################################################################################################################


class dataset_builder():
    ''' 
    Class that iterates over input videos and builds dataset
            
    In order to make proceed the code,
    make sure, you have next files in your working directory:

        folder "face-alignment with files and folders inside"
        path2data example:  "/home/Q/lipread_mp4/"
        path2model example: "/home/Q/face-alignment/face_alignment/"
        
        All data will be stored in self. :   x_train, y_train
                                             x_val, y_val
                                             x_test, y_test
        
    '''
    def __init__(self, path, gpu = True):
        self.labels = get_labels(path)
        self.path = path
        
        # init of pretrained model
        self.detector_dlib = dlib.get_frontal_face_detector()
        self.predictor_dlib = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def get_train(self, examples_per_label, label_start_index, AT = False):
        ''' 
        Main functions of class. 
            
        N: number of examples [up to 800] from each label [500] you want to preprocess
        AT: True, if you want to add augmented (+- 20 degrees rotated images)
        label_start_index: in order to speed up dataset building process, it was decided
        to use multiple machines
        
        
        Returns: dataset in form of [{Nx500} x 29 x 112 x 112]
                 of cropped mouth regions from videos
                 as well as calculates total time taken
                 
        '''
        start = time.time()
        hdf5_idx = 0
        N = 40000
        stop = N-1 
        self.x_train= np.zeros((N,29,112,112), dtype = np.uint8)
        self.y_train = np.zeros((N,29,1), dtype = np.uint16)
        i_0 = 0
        for label, value in self.labels[label_start_index:]:
            p4g = self.path + "/" + label + "/train/" + "*.mp4"
            video_files = glob.glob(p4g)
            video_files = video_files[:examples_per_label]
            i_1 = len(video_files) + i_0
            self.y_train[i_0:i_1] = value
            for ix, video in enumerate(video_files):
                result = video2data(video, self.detector_dlib, self.predictor_dlib, False)
                index = i_0 + ix
                if type(result) == np.ndarray:
                    self.x_train[index] = result
                    
            if index >= stop:
                hdf5_path = '/home/Q/' + str(hdf5_idx) + 'part.hdf5'
                hdf5_saver(self.x_train, self.y_train, hdf5_path, 'x_train', 'y_train')
                hdf5_idx += 1
                i_0 = 0
                self.x_train = np.zeros((N,29,112,112), dtype = np.uint8)
                self.y_train = np.zeros((N,1), dtype = np.uint16)
                print (start - time.time())
            i_0 = i_1 
        
              
    def get_val(self, examples_per_label, hdf5_path, AT = False):
        ''' 
        SEE API get_train
        examples_per_label maximum value is 50
        hdf5_path example: "/home/USERNAME/val_set.hdf5"
        '''    
        start = time.time()
        N = examples_per_label * 500
        stop = N-1 # last index of each label
        self.x_val= np.zeros((N,29,112,112), dtype = np.uint8)
        self.y_val = np.zeros((N,29,1), dtype = np.uint16)
        i_0 = 0
        for label, value in self.labels.items():
            p4g = self.path + "/" + label + "/val/" + "*.mp4"
            video_files = glob.glob(p4g)
            video_files = video_files[:examples_per_label]
            i_1 = len(video_files) + i_0
            self.y_val[i_0:i_1] = value
            for ix, video in enumerate(video_files):
                result = video2data(video, self.detector_dlib, self.predictor_dlib, False)
                index = i_0 + ix
                if type(result) == np.ndarray:
                    self.x_val[index] = result  
            if index >= stop:
                hdf5_saver(self.x_val, self.y_val, hdf5_path, 'x_val', 'y_val')
                print (start - time.time())
            i_0 = i_1 
         
        
        
    def get_test(self, examples_per_label, hdf5_path, AT = False):
        ''' 
        SEE API get_train
        examples_per_label maximum value is 50
        hdf5_path example: "/home/USERNAME/test_set.hdf5"
        '''    
        start = time.time()
        N = examples_per_label * 500
        stop = N-1 # last index of each label
        self.x_test= np.zeros((N,29,112,112), dtype = np.uint8)
        self.y_test = np.zeros((N,29,1), dtype = np.uint16)
        i_0 = 0
        for label, value in self.labels.items():
            p4g = self.path + "/" + label + "/train/" + "*.mp4"
            video_files = glob.glob(p4g)
            video_files = video_files[:examples_per_label]
            i_1 = len(video_files) + i_0
            self.y_test[i_0:i_1] = value
            for ix, video in enumerate(video_files):
                result = video2data(video, self.detector_dlib, self.predictor_dlib, False)
                index = i_0 + ix
                if type(result) == np.ndarray:
                    self.x_test[index] = result  
            if index >= stop:
                hdf5_saver(self.x_test, self.y_test, hdf5_path, 'x_test', 'y_test')
                print (start - time.time())
            i_0 = i_1 
      