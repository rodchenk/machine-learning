# machine-learning
Get started with machine learning and TensorFlow (tf v.2.1.0, Python v.3.7.6, Anaconda v.4.7.12, pip v.20.0.2)

### Set and manage virtual environment

```bash
$ py -m pip install --upgrade pip
$ py -m pip install --user virtualenv
$ py -m venv env
$ .\env\Scripts\activate
(env) pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
(env) deactivate
```
### Or set venv with Anaconda

```bash
$ conda create -n tensorflow_env tensorflow
$ conda activate tensorflow_env
$ conda deactivate
