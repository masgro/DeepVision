
from efficientnet.tfkeras import EfficientNetB0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger,TensorBoard

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import os

# W&B Imports
#import wandb
#from wandb.keras import WandbCallback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# This is secret and shouldn't be checked into version control
#WANDB_API_KEY="c329d6367775bffc964f667cec47f1b912d97508"

# Name and notes optional
#WANDB_NAME="DeepVsion"
#WANDB_NOTES="Smaller learning rate, more regularization."

#wandb.init(project="deepvision-final-work",config={"hyper": "parameter"})

print(tf.__version__)
print("GPU AVAILABLE: ",tf.test.is_gpu_available())

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # making sure Tensorflow doesn't overflow the GPU

# Some constants
IMG_ROWS = 256
IMG_COLS = 256
NUM_CLASSES = 17
TEST_SIZE = 0.2
RANDOM_STATE = 137

#Model
NO_EPOCHS = 30
BATCH_SIZE = 32

##################################
## THE DATA ######################
##################################

filename = "train-jpg-labels.pkl" 
etiqueta = pd.read_pickle(filename)

etiqueta.dtypes

Nimages = etiqueta["image_name"].size
print(Nimages)

train, test = train_test_split(etiqueta, test_size = TEST_SIZE, random_state = RANDOM_STATE)
print(train["image_name"].size)
print(test["image_name"].size)

# Training data generator
datagen_train = ImageDataGenerator(
    rescale=1./255,  
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range = 45,
    width_shift_range=0.2,
    height_shift_range=0.2)

# Validation data generator
datagen_val = ImageDataGenerator(
    rescale=1./255)


##################################
## MODEL #########################
##################################

input_shape = (IMG_ROWS,IMG_COLS,3)

#### Efficient ####
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

effnet.trainable = True

x = effnet.output
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs = effnet.input, outputs = predictions)
###################


#### ResNet ####
#restnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
#
#output = restnet.layers[-1].output
#output = Flatten()(output)
#restnet = Model(inputs=restnet.input,outputs=output)
#
#for i,layer in enumerate(restnet.layers):
#    layer.trainable = False
#
#model = Sequential()
#
#model.add(restnet)
#
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
#
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
#
#model.add(Dense(NUM_CLASSES, activation='sigmoid'))
################

METRICS = [keras.metrics.TruePositives(name="tp"),
           keras.metrics.FalsePositives(name="fp"),
           keras.metrics.TrueNegatives(name="tn"),
           keras.metrics.FalseNegatives(name="fn"),
           keras.metrics.BinaryAccuracy(name="accuracy"),
           keras.metrics.Precision(name="precision"),
           keras.metrics.Recall(name="recall"),
           keras.metrics.AUC(name="auc")]

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5,decay=1.0e-6),
              metrics=METRICS)

#model.load_weights("best_model_with_resnet_checkpoint.hdf5")

##################################
## CALLBACKS #####################
##################################

earlystopping = EarlyStopping(monitor='val_loss', 
                              min_delta = 0, 
                              patience = 5, 
                              verbose = 1, 
                              mode = 'auto', 
                              baseline = None, 
                              restore_best_weights = False)

check_point_file = "checkpoint_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5"
checkpoint = ModelCheckpoint(check_point_file,
                             monitor = 'loss',
                             verbose = 0,
                             save_best_only = True,
                             mode = 'auto',
                             save_freq = 'epoch')

csvlogger = CSVLogger("training.log")

#wandbcallback = WandbCallback(save_model=False,
#                              monitor="val_loss",
#                              mode='auto',
#                              data_type=None,
#                              validation_data=None,
#                              predictions=8,
#                              generator=None)

log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, 
                          histogram_freq = 1,
                          write_graph = True,
                          write_images = False,
                          update_freq = 'epoch')

callbacks = [checkpoint,csvlogger,tensorboard]#wandbcallback]

##################################
## TRAINING ######################
##################################

directory = ""
x_col = "image_name"
y_col = "tags"

keras.backend.clear_session()  # For easy reset of notebook state.

# Train!
train_model = model.fit_generator(
    datagen_train.flow_from_dataframe(train,
                    directory=directory,
                    x_col=x_col, 
                    y_col=y_col, 
                    weight_col=None, 
                    target_size=(IMG_ROWS, IMG_COLS), 
                    color_mode='rgb', 
                    classes=None, 
                    class_mode='categorical', 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='jpg',
                    subset=None,
                    interpolation='nearest',
                    validate_filenames=False),
    validation_data=datagen_val.flow_from_dataframe(test,
                    directory=directory,
                    x_col=x_col, 
                    y_col=y_col, 
                    weight_col=None, 
                    target_size=(IMG_ROWS, IMG_COLS), 
                    color_mode='rgb', 
                    classes=None, 
                    class_mode='categorical', 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='jpg',
                    subset=None,
                    interpolation='nearest',
                    validate_filenames=False),
    epochs = NO_EPOCHS,
    steps_per_epoch = train.size // BATCH_SIZE,
    validation_steps = test.size // BATCH_SIZE,
    callbacks = callbacks)

model.save_weights("last_weights.hdf5",overwrite=True,save_format="h5")
