
import csv
import os
import shutil
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Setup
directories = ['train', 'test', 'validate']
categories = ['reservoirs', 'terrains']
augment_n = [5, 5]

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

for dir in directories:
    for j in range(len(categories)):
        cat = categories[j]
        n = augment_n[j]
        path = dir + '/' + cat
        image_list = os.listdir(path)
        for file in image_list:
            img = load_img(path = (path + '/' + file),  target_size=(224, 224))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=(dir + '/' + cat), save_prefix='aug_', save_format='TIF'):
                i += 1
                if i >= n:
                    break