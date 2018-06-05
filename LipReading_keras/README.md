# Lipreading With Machine Learning In keras
A keras implementation of the models described in [Combining Residual Networks with LSTMs for Lipreading]  by T. Stafylakis and G. Tzimiropoulos.

## Usage
 - Install [Python 3].
 - Clone the repository.
 - Run `pip3 install -r requirements.txt` to install project dependencies.
 - download shape_predictor_68_face_landmarks.dat from  <http://dlib.net/files/>
 - Build the data with `python3 buildDataset.py` 
 - to use, run  `python3 main.py`.

## Dependencies
 - [keras]
 - [Python 3] to run the program
 - [tensorflow] for tensors
 - [dlib] for face detection model
 - [NumPy] to visualize individual layers

   [Combining Residual Networks with LSTMs for Lipreading]: <https://arxiv.org/pdf/1703.04105.pdf>
   [Python 3]: <https://www.python.org/downloads/>
   [tensorflow]: <https://www.tensorflow.org/>
   [keras]: <https://keras.io/>
   [dlib]: <http://dlib.net/>
   [NumPy]: <http://www.numpy.org/>
   
   
