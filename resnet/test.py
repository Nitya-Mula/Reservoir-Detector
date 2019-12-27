import numpy as np
import keras
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

valid_batches = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory('validate', class_mode='binary', classes=['terrains', 'reservoirs'], batch_size=50, target_size=(224, 224), shuffle=False)
model = load_model('trained_resnet.h5')
model.compile(SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

_, acc = model.evaluate_generator(valid_batches, steps=len(valid_batches), verbose=0)
print('Accuracy is :' + str(acc * 100))
valid_batches.reset()
prediction = model.predict_generator(valid_batches, steps=len(valid_batches))

y_true = np.array([0]*25 + [1]*25)
y_pred = prediction > 0.5

cm = confusion_matrix(y_true, y_pred)

labels = ['Terrains', 'Reservoirs']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
	for j in range(cm.shape[1]):
		ax.text(j, i, cm[i, j],
				ha="center", va="center",
				color="white" if cm[i, j] > thresh else "black")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
