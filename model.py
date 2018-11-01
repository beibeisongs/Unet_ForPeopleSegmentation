# encoding=utf-8
# Date: 2018-10-18

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)  # Note: inputs = Tensor("input_1:0", shape=(?, 256, 256, 1), dtype=float32)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)    # Note: conv1 = Tensor("conv2d_1/Relu:0", shape=(?, 256, 256, 64), dtype=float32); 64 is the num of the filters; 3 is the size of the filters
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)     # Note: conv1 = Tensor("conv2d_2/Relu:0", shape=(?, 256, 256, 64), dtype=float32); Be careful about the name of the variable while debugging
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)   # Note: pool1 = Tensor("max_pooling2d_1/MaxPool:0", shape=(?, 128, 128, 64), dtype=float32)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)    # Note: conv2 = Tensor("conv2d_3/Relu:0", shape=(?, 128, 128, 128), dtype=float32)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)    # Note: conv2 = Tensor("conv2d_4/Relu:0", shape=(?, 128, 128, 128), dtype=float32); Be careful about the name of the variable while debugging
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)   # Note: pool2 = Tensor("max_pooling2d_2/MaxPool:0", shape=(?, 64, 64, 128), dtype=float32)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)    # Note: conv3 = Tensor("conv2d_5/Relu:0", shape=(?, 64, 64, 256), dtype=float32)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)    # Note: conv3 = Tensor("conv2d_6/Relu:0", shape=(?, 64, 64, 256), dtype=float32); Be careful about the name of the variable while debugging
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)   # Note: pool3 = Tensor("max_pooling2d_3/MaxPool:0", shape=(?, 32, 32, 256), dtype=float32)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)    # Note: Tensor("conv2d_7/Relu:0", shape=(?, 32, 32, 512), dtype=float32)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)    # Note: Tensor("conv2d_8/Relu:0", shape=(?, 32, 32, 512), dtype=float32)
    drop4 = Dropout(0.5)(conv4)    # Note: Tensor("dropout_1/cond/Merge:0", shape=(?, 32, 32, 512), dtype=float32)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)    # Note: Tensor("max_pooling2d_4/MaxPool:0", shape=(?, 16, 16, 512), dtype=float32)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)    # Note: Tensor("conv2d_9/Relu:0", shape=(?, 16, 16, 1024), dtype=float32)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)    # Note: Tensor("conv2d_10/Relu:0", shape=(?, 16, 16, 1024), dtype=float32)
    drop5 = Dropout(0.5)(conv5)    # Note: Tensor("dropout_2/cond/Merge:0", shape=(?, 16, 16, 1024), dtype=float32)

    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(drop5)    # Note: Tensor("conv2d_transpose_1/Relu:0", shape=(?, ?, ?, 512), dtype=float32)
    merge6 = concatenate([drop4, up6], axis=3)    # Note: Tensor("concatenate_1/concat:0", shape=(?, 32, 32, 1024), dtype=float32)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)    # Note: Tensor("conv2d_11/Relu:0", shape=(?, 32, 32, 512), dtype=float32)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)    # Note: Tensor("conv2d_12/Relu:0", shape=(?, 32, 32, 512), dtype=float32)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv6)    # Note: Tensor("conv2d_transpose_2/Relu:0", shape=(?, ?, ?, 256), dtype=float32)
    merge7 = concatenate([conv3, up7], axis=3)    # Note: Tensor("concatenate_2/concat:0", shape=(?, 64, 64, 512), dtype=float32)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)    # Note: Tensor("concatenate_2/concat:0", shape=(?, 64, 64, 512), dtype=float32)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)    # Note: Tensor("conv2d_14/Relu:0", shape=(?, 64, 64, 256), dtype=float32)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv7)    # Note:
    merge8 = concatenate([conv2, up8], axis=3)    # Note: Tensor("conv2d_transpose_3/Relu:0", shape=(?, ?, ?, 128), dtype=float32)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)    # Note: Tensor("conv2d_15/Relu:0", shape=(?, 128, 128, 128), dtype=float32)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)    # Note: Tensor("conv2d_16/Relu:0", shape=(?, 128, 128, 128), dtype=float32)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv8)    # Note: Tensor("conv2d_transpose_4/Relu:0", shape=(?, ?, ?, 64), dtype=float32)
    merge9 = concatenate([conv1, up9], axis=3)    # Note: Tensor("concatenate_4/concat:0", shape=(?, 256, 256, 128), dtype=float32)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)    # Note: Tensor("conv2d_17/Relu:0", shape=(?, 256, 256, 64), dtype=float32)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)    # Note: Tensor("conv2d_18/Relu:0", shape=(?, 256, 256, 64), dtype=float32)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)    # Note: Tensor("conv2d_19/Relu:0", shape=(?, 256, 256, 2), dtype=float32)

    """ Attention: There is a big change, and the original codes is: conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)    # Note: Tensor("conv2d_20/Sigmoid:0", shape=(?, 256, 256, 1), dtype=float32)"""
    conv10 = Conv2D(2, 1, activation='sigmoid')(conv9)    # Note: Tensor("conv2d_20/Sigmoid:0", shape=(?, 256, 256, 2), dtype=float32)

    model = Model(input=inputs, output=conv10)  # Note: inputs = Tensor("input_1:0", shape=(?, 256, 256, 1), dtype=float32); conv10 = Tensor("conv2d_20/Sigmoid:0", shape=(?, 256, 256, 1), dtype=float32)

    # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=RMSprop(lr=2e-5), loss='mse', metrics=['acc'])
    
    model.summary()

    if (pretrained_weights):
        print("Attention: continue training !")
        model.load_weights(pretrained_weights)

    return model

""" Description: 

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 256, 256, 1)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 256, 256, 64) 640         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 256, 256, 64) 36928       conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 128, 128, 64) 0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 128, 128, 128 73856       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 128, 128, 128 147584      conv2d_3[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 64, 64, 128)  0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 64, 256)  295168      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 64, 256)  590080      conv2d_5[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 32, 32, 256)  0           conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 512)  1180160     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 512)  2359808     conv2d_7[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 32, 512)  0           conv2d_8[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 16, 16, 512)  0           dropout_1[0][0]                  
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 1024) 4719616     max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 1024) 9438208     conv2d_9[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 16, 16, 1024) 0           conv2d_10[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 32, 32, 512)  2097664     dropout_2[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 32, 32, 1024) 0           dropout_1[0][0]                  
                                                                 conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 512)  4719104     concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 512)  2359808     conv2d_11[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 64, 64, 256)  524544      conv2d_12[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 64, 64, 512)  0           conv2d_6[0][0]                   
                                                                 conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 64, 256)  1179904     concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 64, 64, 256)  590080      conv2d_13[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 128, 128, 128 131200      conv2d_14[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 128, 128, 256 0           conv2d_4[0][0]                   
                                                                 conv2d_transpose_3[0][0]         
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 128, 128, 128 295040      concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 128, 128 147584      conv2d_15[0][0]                  
__________________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, 256, 256, 64) 32832       conv2d_16[0][0]                  
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 256, 256, 128 0           conv2d_2[0][0]                   
                                                                 conv2d_transpose_4[0][0]         
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 256, 256, 64) 73792       concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 256, 256, 64) 36928       conv2d_17[0][0]                  
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 256, 2)  1154        conv2d_18[0][0]                  
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 256, 256, 1)  3           conv2d_19[0][0]                  
==================================================================================================
Total params: 31,031,685
Trainable params: 31,031,685
Non-trainable params: 0
"""