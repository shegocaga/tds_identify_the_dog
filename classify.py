
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.layers.normalization import batch_normalization 

#import other packages 

import os

import tensorflow as tf
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical # to convert to one-hot encodings
from PIL import Image

from resizeimage import resizeimage

from extract_accuracy import compute_accuracy_metrics
import io
import time
from contextlib import redirect_stdout

#########################################################
#resize images

#Get a list of the image filenames
filenames = glob.glob('/Users/Shegocaga/Documents/Official Dataset/train/*.jpg', recursive = True)

for file in range (len(filenames)):
    resize_image = resizeimage.resize_cover(Image.open(filenames[file]), [100,100])
    resize_image.save(filenames[file])


#########################################################
#evaluate model
def evaluate_model(model, x_test, y_test):
   
    #get index of max value for each row of probablties (high score reflects best guess for the image class)
    y_est = np.argmax(model.predict(x_test),axis=1)
    #get index of max value for y predictor
    y_tru = np.argmax(y_test, axis=1)
    #compute confustion matrix
    cf_result = confusion_matrix(y_tru,y_est)

    correct_hit = 0

    for type in range(y_test.shape[1]-1):
        correct_hit = correct_hit + cf_result[type,type]

    acc = correct_hit/len(y_est)

    return acc, cf_result

#########################################################
#Prepare the data 

#Get a list of the image filenames
filenames = glob.glob('/Users/Shegocaga/Documents/Official Dataset/train/*.jpg', recursive = True)

x = np.array([np.array(plt.imread(filename)) for filename in filenames])
#reshape image array in the tensorflow format
x = np.reshape(x,[x.shape[0],x.shape[1],x.shape[2],x.shape[3]])



y = pd.read_csv('/Users/Shegocaga/Documents/Official Dataset/labels.csv', sep = ',')

breeds = sorted(list(set(y['breed'].values)))
b2id = dict((b,i) for i,b in enumerate(breeds))
breed_vector = [b2id[i] for i in y['breed'].values]
y = to_categorical(breed_vector)


#########################################################
#The Model

def build_model(amt_filters,filter_size,stride):

    acc = Accuracy()

	#Start with a layer that inputs the image data
    #network = input_data(shape=[None, 480,640, 3], data_augmentation = img_aug )
    network = input_data(shape=[None, 100,100, 3] )

    #Add Convolutional Layer to expand the feature space
    network = conv_2d(network, amt_filters, filter_size, strides = stride, activation='relu', name = 'conv_1')
    #Add a pooling layer to reduce the feature space 
    network = max_pool_2d(network, 2, strides=4, name = 'max_1')

    #Add Convolutional Layer to expand the feature space
    network = conv_2d(network, amt_filters, filter_size, strides = stride, activation='relu', name = 'conv_2')
    #Add a pooling layer to reduce the feature space 
    network = max_pool_2d(network, 2, strides=4, name = 'max_2')

    # Add a fully connected layer  
    network = fully_connected(network, 960, activation='elu')

    # Add a fully connected layer  
    network = fully_connected(network, 480, activation='elu')
    
    
    # Add a fully connected layer  
    network = fully_connected(network, 240, activation='elu')
    
    #Add batch normalization
    network = batch_normalization(network)
    
    # add a Dropout layer
#    network = dropout(network, 0.5)
#    # Fully Connected Layer
    network = fully_connected(network, 120, activation='softmax')
    # Final network
    network = regression(network, optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=0.001, metric=acc)


    # The model with details on where to save
    model = tflearn.DNN(network,tensorboard_verbose=0 )

    return model

#########################################################
#Fit the Model 

filter_number =  6
filter_size   =  4
strides       =  2
epoch_number  = 150

#split the data set into test and train sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.10)

#reset the graph (Necceary to loop thorugh momentumtiple fittings; dont know why)
tf.reset_default_graph()
#build the model
start_time = time.clock()
model = build_model(filter_number, filter_size, strides)
#fit the model
model.fit(x_train,y_train,n_epoch=epoch_number, show_metric=True)


finish_time = time.clock()
print("Analysis Time: " + str(round(finish_time-start_time,3)) + " seconds")

acc_test, cf_result = evaluate_model(model,x_test,y_test)
acc_train, cf_result = evaluate_model(model,x_train,y_train)

###############################################################
# #parameter search for optimal response

amt_filters = np.round(np.linspace(2,12,6))
filter_sizes = np.round(np.linspace(2,10,5))
strides = np.round(np.linspace(2,2,1))
 
epoch_number = 150

#split the data set into test and train sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.10)

output = []

for filter_amt in range(amt_filters.shape[0]):
    for filter_size in range(filter_sizes.shape[0]):
        for stride in range(strides.shape[0]):

            #reset the graph (Necceary to loop thorugh momentumtiple fittings; dont know why)
            tf.reset_default_graph()

            print("Fitting Model")
            print("*****")
            print("Amount of Filters: " + str(amt_filters[filter_amt]))
            print("Size of Filter: " + str(filter_sizes[filter_size]))
            print("Strides: " + str(strides[stride]))
            
            start_time = time.clock()
            
            with io.StringIO() as buf, redirect_stdout(buf):
                model = build_model(amt_filters[filter_amt], int(filter_sizes[filter_size]), int(strides[stride]))
                model.fit(x,y, show_metric=True, n_epoch = epoch_number, run_id="testing") 

            finish_time = time.clock()
            acc_test, cf_result = evaluate_model(model,x_test,y_test)
            acc_train, cf_result = evaluate_model(model,x_train,y_train)

            values = [amt_filters[filter_amt], filter_sizes[filter_size], strides[stride], acc_test,acc_train, round(finish_time-start_time,3)]
            print(values)
            output.append(values)
    


