from __future__ import absolute_import, division, print_function, unicode_literals
from matplot import plot_image, plot_value_array, show_simple_pic
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

import os
import glob
import math
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

FASHION_FEATURES = ['Футболка', "Шорты", "Свитер", "Платье", "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]
FASHION_FEATURES_ENG = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
BATCH_SIZE = 32
SHOULD_RESTORE = True

def normalize(images, labels):
  	images = tf.cast(images, tf.float32)
  	images /= 255
  	return images, labels

def mirror(images, labels):
	return tf.image.flip_up_down(images), labels

def simple_display(dataset):
	for image, label in dataset.take(1):
		print('This is %s'%FASHION_FEATURES[label.numpy()])
		show_simple_pic(image)

def model_acc(model, dataset, message=''):
	test_loss, test_accuracy = model.evaluate(dataset, verbose=2)
	print("\n%s accuracy on test dataset: %04.2f\n"%(message, float(100*test_accuracy) ))

def get_chkpath():
	checkpoint_path = "models/fashion_mnist/fashion_mnist.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	return checkpoint_path

def create_model():
	model = tf.keras.Sequential()

	model.add( tf.keras.layers.Conv2D(32, (3,3), padding='same', data_format=None, activation=tf.nn.relu, input_shape=(28, 28, 1)) )
	model.add( tf.keras.layers.MaxPooling2D((2, 2), strides=2) )
	model.add( tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu) )
	model.add( tf.keras.layers.MaxPooling2D((2, 2), strides=2) )
	model.add( tf.keras.layers.Flatten() )
	model.add( tf.keras.layers.Dense(units=128, activation=tf.nn.relu) )
	model.add( tf.keras.layers.Dense(units=10, activation=tf.nn.softmax) )

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model

def restore_and_evaluate(model, test_dataset):
	model_acc(model, test_dataset, message='Untrained model')
	new_model = tf.keras.models.load_model('models/fashion.h5')
	model_acc(new_model, test_dataset, message='Restored model') #getting model accuracy loading weights

def save_hdf5(model):
	model.save('models/fashion.h5')

def train_and_evaluate(model, train_dataset, test_dataset, num_train_examples):
	# no needed anymore, we use h5 format to store the model (callbacks=[cp_callback])
	# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=get_chkpath(), save_weights_only=True, verbose=1) 
	model.fit(train_dataset, epochs=2, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
	save_hdf5(model)

	model_acc(model, test_dataset)

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def evaluate_custom_data():
	model = tf.keras.models.load_model('models/fashion.h5')
	
	image_file_1 = rgb2gray(mpimg.imread('custom_data/10_s28.png')).reshape(28, 28, 1)

	dataset = tf.data.Dataset.from_tensor_slices((tf.constant([image_file_1]), tf.constant([1])))
	dataset = dataset.batch(1)

	for img, label in dataset.take(1):
		predictions = model.predict(img)
		suggestion = np.argmax(predictions)
		for sug, label in zip(predictions[0], FASHION_FEATURES):
			print('{:.2f}%\t{}'.format(float(sug)*100, label))
		print(FASHION_FEATURES[suggestion])
		show_simple_pic(img)
		break

def predict_images():
	model = tf.keras.models.load_model('models/fashion.h5')

	files = glob.glob('custom_data/*.png')
	images = [rgb2gray(mpimg.imread(x)).reshape(28, 28, 1) for x in files]
	dataset = tf.data.Dataset.from_tensor_slices( (tf.constant(images), tf.constant([0]*10)) )
	dataset = dataset.batch(1)

	labels = []
	for pic, lab in dataset:
		predictions = model.predict(pic)
		suggestion = np.argmax(predictions)
		labels.append(FASHION_FEATURES[suggestion])

	plt.figure(figsize=(10,5))
	for i in range(10):
		plt.subplot(2, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)

		image = np.array(images[i], dtype='float')
		pixels = image.reshape((28, 28))
		plt.imshow(pixels, cmap=plt.cm.binary)
		plt.xlabel(labels[i])
	plt.show()

def __main():
	dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
	train_dataset, test_dataset = dataset['train'].map(normalize), dataset['test'].map(normalize)
	num_train_examples = metadata.splits['train'].num_examples

	model = create_model()
	model.summary()

	train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
	test_dataset = test_dataset.batch(BATCH_SIZE)

	if SHOULD_RESTORE:
		restore_and_evaluate(model=model, test_dataset=test_dataset)
	else:
		train_and_evaluate(model=model, train_dataset=train_dataset, test_dataset=test_dataset, num_train_examples=num_train_examples)
		

if __name__ == '__main__':
	#__main()
	#evaluate_custom_data()
	predict_images()