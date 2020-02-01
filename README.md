<img src="https://seeklogo.com/images/T/tensorflow-logo-AE5100E55E-seeklogo.com.png" width="250" align="right" alt="Tensorflow">

# ML with Tensorflow and Keras
Get started with machine learning and TensorFlow (tf v.2.1.0, Python v.3.7.6, Anaconda v.4.7.12, pip v.20.0.2)

## Installation

Install tensorflow for Python with pip. Detailed information on [Tensorflow](https://www.tensorflow.org/install/pip) site.

### Install tensorflow and set virtual environment

```php
$ py -m pip install --upgrade pip
$ py -m pip install --user virtualenv
$ py -m venv env
$ .\env\Scripts\activate
(env) pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
(env) deactivate

#or with Anaconda

$ conda create -n tensorflow_env tensorflow
$ conda activate tensorflow_env
(env) python -m pip install --upgrade pip
(env) pip install --upgrade tensorflow
(env) conda deactivate
```

### Check if it works
```python
>> import tensorflow as tf
>> print('Tensorflow version: {}'.format(tf.__version__))
```
## Usage

All models are stored in [h5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format. It's easy to restore the model with its weight and variables and start working without having to re-train the model again

### Fashion

The model was trained with fashion_mnist dataset. You can predict own images:

```python
from mnist_fashion import FASHION_FEATURES
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image_file = rgb2gray( mpimg.imread('custom_data/2_s28.png') ).reshape(28, 28, 1)

filenames = tf.constant([image_file])
labels = tf.constant([0])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.batch(BATCH_SIZE)

for img, label in dataset.take(1):
  predictions = model.predict(img)
  suggestion = np.argmax(predictions)
  print(FASHION_FEATURES[suggestion])
  break
```
