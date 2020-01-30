from __future__ import absolute_import, division, print_function, unicode_literals
from matplot import plot_image, plot_value_array, show_simple_pic

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import os
import math
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

FASHION_FEATURES = ['Футболка', "Шорты", "Свитер", "Платье", "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]
BATCH_SIZE = 32
SHOULD_RESTORE = True

def normalize(images, labels):
  	images = tf.cast(images, tf.float32)
  	images /= 255
  	return images, labels

def mirror(images, labels):
	return tf.image.flip_up_down(images), labels

def simple_display(dastaset):
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

	model.load_weights(get_chkpath())
	model_acc(model, test_dataset, message='Restored model') #getting model accuracy loading weights

def train_and_evaluate(model, train_dataset, test_dataset, num_train_examples):
	# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=get_chkpath(), save_weights_only=True, verbose=1)

	model.fit(train_dataset, epochs=2, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE), callbacks=[cp_callback])
	model_acc(model, test_dataset)

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
	__main()