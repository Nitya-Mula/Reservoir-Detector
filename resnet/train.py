import numpy as np
import keras
from keras import backend as keras_beckend
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD, rmsprop
from keras.metrics import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train_batches = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory('train', class_mode='binary', classes=['terrains', 'reservoirs'], batch_size=10, target_size=(224, 224))
test_batches = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory('test', class_mode='binary', classes=['terrains', 'reservoirs'], batch_size=10, target_size=(224, 224))

model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3))
# Mark loaded layers as not trainable
# for layer in model.layers:
#     layer.trainable = False
# Add new classifier layers

flat1 = Flatten(input_shape=model.output_shape[1:])(model.layers[-1].output)
class1 = Dense(256, activation='relu')(flat1)
output = Dropout(0.5)(class1)
output = Dense(1, activation='sigmoid')(output)

# Define new model
model = Model(inputs=model.inputs, outputs=output)
model.summary()

model.compile(SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batches, steps_per_epoch=len(train_batches), validation_data=test_batches, validation_steps=len(test_batches), epochs=10, verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
# Accuracy
#print(history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_accuracy'], loc='upper left')
plt.show()

_, acc = model.evaluate_generator(test_batches, steps=len(test_batches), verbose=0)
print('Accuracy is: ' + str(acc * 100))
model.save('trained_resnet.h5')
