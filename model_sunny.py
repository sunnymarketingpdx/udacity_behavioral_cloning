import pickle
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

#Reload the x data (images) from pickle file
pickle_file_x = 'sample_data_x.p'
with open(pickle_file_x, 'rb') as file_x:
    x_train = pickle.load(file_x)
print ("shape of x training data")
print (np.array(x_train).shape)
x_train = np.array(x_train)

#Reload the y data (steering angles) from pickle file
pickle_file_y = 'sample_data_y.p'
with open(pickle_file_y, 'rb') as file_y:
    y_train = pickle.load(file_y)
print ("number of y training data")
print (len(y_train))
y_train = np.array(y_train)

#shufle both x and y data
x_train, y_train = shuffle(x_train, y_train)

# This is the shape of the image
input_shape = x_train.shape[1:]
print(input_shape, 'input shape')
# Set the parameters and print out the summary of the model
#np.random.seed(1337)  # for reproducibility

batch_size = 64 # The lower the better
nb_classes = 1 # The output is a single digit: a steering angle
nb_epoch = 10 # The higher the better

# import model and weights if exists
#try:
	#with open('model.json', 'r') as jfile:
	    #model = model_from_json(json.load(jfile))

	# Use adam and mean squared error for training
	#model.compile("adam", "mse")

	# import weights
	#model.load_weights('model.h5')

	#print("Imported model and weights")

# If the model and weights do not exist, create a new model
#except:
	# If model and weights do not exist in the local folder,
	# initiate a model

	# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

# Initiating the model
model = Sequential()

# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1], border_mode='valid',input_shape=input_shape))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
	# Applying ReLU
model.add(Activation('relu'))
	# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
	# Applying ReLU
model.add(Activation('relu'))
	# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
	# Apply dropout of 25%
model.add(Dropout(0.25))

	# Flatten the matrix. The input has size of 360
model.add(Flatten())
	# Input 360 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Input 16 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Input 16 Output 16
model.add(Dense(16))
	# Applying ReLU
model.add(Activation('relu'))
	# Apply dropout of 50%
model.add(Dropout(0.5))
	# Input 16 Output 1
model.add(Dense(nb_classes))

# Print out summary of the model
model.summary()

# Compile model using Adam optimizer
# and loss computed by mean squared error
model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['accuracy'])

### Model training
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20)
history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_validation, y_validation))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

import json
import os
import h5py

# Save the model.
# If the model.json file already exists in the local file,
# warn the user to make sure if user wants to overwrite the model.
#if 'model.json' in os.listdir():
	#print("The file already exists")
	#print("Want to overwite? y or n")
	#user_input = input()

	#if user_input == "y":
		# Save model as json file
		#json_string = model.to_json()

		#with open('model.json', 'w') as outfile:
			#json.dump(json_string, outfile)

			# save weights
			#model.save_weights('./model.h5')
			#print("Overwrite Successful")
	#else:
		#print("the model is not saved")
#else:
	# Save model as json file

model.save("model.h5")
#json_string = model.to_json()

#with open('model.json', 'w') as outfile:
	 #json.dump(json_string, outfile)

		# save weights
#model.save_weights('./model.h5')
print("Saved")
