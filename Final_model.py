from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Tensorflow/keras code modified by L.O. Hall 4/8/19 to load 5 of 6 animal classes for training (no frog)

import keras
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler

import os
import math

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
#from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export

batch_size = 64
num_classes = 10
epochs = 125
data_augmentation = True
#data_augmentation = False
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


def load_data5():
  """Loads CIFAR10 dataset. However, just 5 classes, all animals except frog
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
#  dirname = 'cifar-10-batches-py'
#  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#  path = get_file(dirname, origin=origin,  untar=True)
  path= '/data/DrHallClassData/cifar-10-batches-py'
  num_train_samples = 50000
  num_5_class = 25000
  num_5_test = 5000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')
  x5_train = np.empty((num_5_class, 32, 32, 3), dtype='uint8')
  y5_train = np.empty((num_5_class,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
#    print('Here')
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    # find train data of interest
  count=0

  for i in range(0, len(y_train)-1):
   if (y_train[i][0] == 2) or (y_train[i][0] == 3) or (y_train[i][0] == 4) or (y_train[i][0] == 5) or (y_train[i][0] == 7):
    x5_train[count]=x_train[i]
    y5_train[count]=y_train[i]
    count=count+1
    
    # find test data of interest
  count=0
  x5_test=np.empty((num_5_test, 32, 32, 3), dtype='uint8')
  y5_test= np.empty((num_5_test,1))

  for i in range(0, len(y_test)-1):
   if (y_test[i][0] == 2) or (y_test[i][0] == 3) or (y_test[i][0] == 4) or (y_test[i][0] == 5) or (y_test[i][0] == 7):
    x5_test[count]=x_test[i]
    y5_test[count]=y_test[i]
    count=count+1
    
#  return (x_train, y_train), (x_test, y_test)
  return (x5_train, y5_train), (x5_test, y5_test)


# Changing the learning rate over epoch.
# This is a good practice to ensure a stable model and it might increase the accuracy.
# Reference: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
def lr_schedule(e):
    l_rate = 0.001
    if e > 75:
        l_rate = 0.0005
    if e > 100:
        l_rate = 0.0003
    return l_rate

# Showing the result of the prediction and the true classification.
# Reference: http://parneetk.github.io/blog/cnn-cifar10/
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    p_class = np.argmax(result, axis=1)
    t_class = np.argmax(test_y, axis=1)
    return (p_class, t_class)


# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = load_data5()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# Calculate the z-score of the prediction.
# Reference: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)


steps_for_epoch = math.ceil(x_train.shape[0] / batch_size)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Adding new Convolutional layers and Pooling layers with regularization and BatchNormalization.
# Reference: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    
    
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch = steps_for_epoch,
                        epochs=epochs, verbose =1,
                        validation_data=(x_test, y_test),
                        callbacks = [LearningRateScheduler(lr_schedule)])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Export the classification to csv file.
x, y = accuracy(x_train, y_train, model);
print('x_train: ', x);
print('y_train: ', y);
np.savetxt("predicted_train.csv", x, delimiter=",")
np.savetxt("true_train.csv", y, delimiter=",")

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Export the classification to csv file.
x, y = accuracy(x_test, y_test, model);
print('x_test: ', x);
print('y_test: ', y);
np.savetxt("predicted_test.csv", x, delimiter=",")
np.savetxt("true_test.csv", y, delimiter=",")
