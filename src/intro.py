import tensorflow.compat.v1 as tf

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.disable_v2_behavior()

	x1 = tf.constant([2,2,3,4])
	x2 = tf.constant([5,6,7,8])

	result = tf.multiply(x1, x2)
	sess = tf.Session()

	try:
		print(sess.run(result))
	finally:
		sess.close()