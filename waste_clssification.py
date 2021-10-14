import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import MobileNet, MobileNetV2, DenseNet121, NASNetMobile, EfficientNetB0, EfficientNetB1
from tensorflow.keras.activations import *
from tensorflow.keras.losses import *
from keras.callbacks import LearningRateScheduler

import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import joblib
from keras.preprocessing.image import load_img
from keras.preprocessing import image

import warnings
warnings.filterwarnings("ignore")

FILEPATH= 'Waste-Dataset/'

print(tf.executing_eagerly())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
EPOCH = 10

train_dir = 'Waste-Dataset/train/'
train_datagen = ImageDataGenerator( rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=8,
                                                    class_mode='categorical',
                                                    target_size=(224, 224),
                                                    shuffle = True)

validation_dir = 'Waste-Dataset/test/'
validation_datagen = ImageDataGenerator( rescale=1.0/255,
    horizontal_flip=True,
    vertical_flip=True)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=8,
                                                              class_mode ='categorical',
                                                              target_size=(224, 224),
                                                              shuffle = True)

K.clear_session()
base_model = MobileNet(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(224,224,3), classes = 2
)

for layer in base_model.layers:
  layer.trainable =  True

def make_model():
  inp= Input(shape=(224, 224, 3))
  if(len(inp.shape)==3):
    inp = tf.expand_dims(inp, axis=0)
  x = base_model(inp)
  x= Flatten()(x)
  x= Dropout(0.3)(x)
  x= Dense(68)(x)
  x= Dense(34,activation='softmax')(x)
  model = tf.keras.Model(inputs= inp, outputs= x)
  return model


# with strategy.scope():
model = make_model()
model.load_weights('waste-classifier.hdf5')
model.compile(optimizer = Adam(learning_rate=2e-4), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

filepath= FILEPATH + 'waste-classifier1.hdf5'  
my_callbacks = [
  ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=True,
    mode='min', min_delta=0.01, cooldown=0, min_lr=0 
  ), 
  ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode= 'max')]

for layer in model.layers:
  layer.trainable=True


fine_tune_at = int (len(model.layers)*0.9)
for layer in model.layers[:fine_tune_at]:
  layer.trainable =  False

history = model.fit(train_generator,
                    shuffle = True,
                    epochs=int(EPOCH*1.5),
                    verbose=True,
                    use_multiprocessing=True,
                    validation_data=validation_generator,
                    callbacks=my_callbacks)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc)
print(val_acc)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

for layer in model.layers:
  layer.trainable=True


fine_tune_at = int (len(model.layers)*0.7)
for layer in model.layers[:fine_tune_at]:
  layer.trainable =  False

history = model.fit(train_generator,
                    shuffle = True,
                    epochs=int(EPOCH*2),
                    verbose=True,
                    use_multiprocessing=True,
                    validation_data=validation_generator,
                    callbacks=my_callbacks)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc)
print(val_acc)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

for layer in model.layers:
  layer.trainable=True


fine_tune_at = int (len(model.layers)*0.6)
for layer in model.layers[:fine_tune_at]:
  layer.trainable =  False

history = model.fit(train_generator,
                    shuffle = True,
                    epochs=int(EPOCH),
                    verbose=True,
                    use_multiprocessing=True,
                    validation_data=validation_generator,
                    callbacks=my_callbacks)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc)
print(val_acc)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

"""Creating the TFLITE model"""

model = tf.keras.models.load_model('waste-classifier.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY, tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

import pathlib
tflite_model_file = pathlib.Path(FILEPATH +'waste-classifier.tflite')
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_path= 'waste-classifier.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
