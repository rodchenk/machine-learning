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
>> tf.VERSION
```
## Usage

All models are stored in [h5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format. It's easy to restore the model with its weight and variables and start working without having to re-train the model again

### Fashion

The model was trained with fashion_mnist dataset:

```python
@TODO
def __main:
  pass
```
