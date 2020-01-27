# machine-learning
Get started with machine learning and TensorFlow

### Set and manage virtual environment

```bash
$ py -m pip install --upgrade pip
$ py -m pip install --user virtualenv
$ py -m venv env
$ .\env\Scripts\activate
(env) pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
(env) deactivate
```
### With Anaconda

```bash
$ conda create -n tensorflow_env tensorflow
$ conda activate tensorflow_env
$ conda deactivate
