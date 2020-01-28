from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

celsius_q    = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

def loss_graph(history):
	import matplotlib.pyplot as plt
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(history)

if __name__ == '__main__':
	layer = tf.keras.layers.Dense(units=1, input_shape=[1])
	model = tf.keras.Sequential([layer])
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
	history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
	print(model.predict([10.0]))

	#loss_graph(history.history['loss'])