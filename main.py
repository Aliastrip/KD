from __future__ import absolute_import, division, print_function, unicode_literals
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten
from keras.preprocessing import image
# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import os, shutil


def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    "E:\\DRIVE\\Projects\\Current\\KD\\files",
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_gen = test_datagen.flow_from_directory(
    "E:\\DRIVE\\Projects\\Current\\KD\\test",
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = build_model()
his = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=100,
    validation_data=valid_gen,
    validation_steps=50
)

acc = his.history['acc']
val_acc = his.history['val_acc']

loss = his.history['loss']
val_loss = his.history['val_loss']

plt.plot(acc)
plt.plot(val_acc)
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
