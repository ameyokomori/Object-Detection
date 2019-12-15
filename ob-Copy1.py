#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD, RMSprop
from lsuv_init import LSUVinit

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
K.clear_session()
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)


# In[2]:


(x_train,y_train),(x_test,y_test) = cifar100.load_data()


# In[3]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[4]:


plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[1])
plt.show()


# In[5]:


x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.np_utils.to_categorical(y_train, 100)
y_test = keras.utils.np_utils.to_categorical(y_test, 100)


# In[6]:


model = Sequential()


# In[7]:


model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[8]:


model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[9]:


model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[10]:


model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# In[11]:


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))


# In[12]:


model.summary()


# In[13]:


plot_model(model,to_file='model2.png')


# In[14]:


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# In[15]:


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[16]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=10, 
                              verbose=0, 
                              mode='auto', 
                              min_delta=0.0001, 
                              cooldown=0, 
                              min_lr=0
)


# In[17]:


tensorboard = TensorBoard(log_dir='./logs2',
                histogram_freq=0,
                batch_size=64,
                write_graph=True,
                write_grads=False,
                write_images=True,
                embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None
)


# In[18]:


model_checkpoing = ModelCheckpoint(filepath='./tmp_models/weights2.hdf5', 
                                   monitor='val_loss', 
                                   verbose=0, 
                                   save_best_only=True, 
                                   save_weights_only=False, 
                                   mode='auto', 
                                   period=1
)


# In[19]:


earlystop = EarlyStopping(monitor='val_loss', 
                          patience=20, 
                          verbose=0, 
                          mode='auto'
)


# In[20]:


datagen = ImageDataGenerator(rotation_range=20, 
                             zoom_range=0.15, 
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             horizontal_flip=True
)


# In[21]:


datagen.fit(x_train)


# In[ ]:


history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), 
                              steps_per_epoch=x_train.shape[0] // 64, 
                              epochs=300, 
                              validation_data=(x_test, y_test), 
                              workers=4, 
                              verbose=1, 
                              callbacks=[tensorboard, model_checkpoing, reduce_lr]
)


# In[ ]:


model.save('full_model2.h5')


# In[ ]:


fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#
fig.savefig('accuracy2.png')


# In[ ]:


fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
#
fig.savefig('loss2.png')


# In[ ]:




