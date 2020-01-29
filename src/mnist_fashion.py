from __future__ import absolute_import, division, print_function, unicode_literals
from matplot import plot_image, plot_value_array, show_simple_pic

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import math
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

class_names = ['Футболка', "Шорты", "Свитер", "Платье", "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]

def normalize(images, labels):
  	images = tf.cast(images, tf.float32)
  	images /= 255
  	#images = images.reshape(4, 28, 28, 1)
  	return images, labels

def mirror(images, labels):
	return tf.image.flip_up_down(images), labels

def simple_display(dastaset):
	for image, label in dataset.take(1):
		print('This is %s'%class_names[label.numpy()])
		show_simple_pic(image)

def model_acc(model, dataset):
	test_loss, test_accuracy = model.evaluate(dataset)
	print("Accuracy on test dataset: ", test_accuracy)

def __main():
	dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
	train_dataset, test_dataset = dataset['train'], dataset['test']

	num_train_examples = metadata.splits['train'].num_examples
	num_test_examples = metadata.splits['test'].num_examples

	train_dataset = train_dataset.map(normalize)
	test_dataset = test_dataset.map(normalize).map(mirror)

	model = tf.keras.Sequential()

	model.add( tf.keras.layers.Conv2D(32, (3,3), padding='same', data_format=None, activation=tf.nn.relu, input_shape=(28, 28, 1)) )
	model.add( tf.keras.layers.MaxPooling2D((2, 2), strides=2) )
	model.add( tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu) )
	model.add( tf.keras.layers.MaxPooling2D((2, 2), strides=2) )
	model.add( tf.keras.layers.Flatten() )
	model.add( tf.keras.layers.Dense(units=128, activation=tf.nn.relu) )
	model.add( tf.keras.layers.Dense(units=10, activation=tf.nn.softmax) )

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	BATCH_SIZE = 32
	train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
	test_dataset = test_dataset.batch(BATCH_SIZE)

	model.fit(train_dataset, epochs=2, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

	model_acc(model, test_dataset) #getting model accuracy after training

if __name__ == '__main__':
	__main()