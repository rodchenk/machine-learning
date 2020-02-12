## Object detection with Tensorflow. Taken from [Gilbert Tanner](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api)

### Install dependencies

```php
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

### Install COCO API

```php
pip install https://github.com/philferriere/cocoapi.git
cp cocoapi/PythonAPI <tensorflow>/models/research/
```

### Install **Protobuf** from [Website](https://github.com/protocolbuffers/protobuf/releases)

Then copy `proto`-folder to \<tensorflow>/research and run the following command to compile all `.proto`-files to python:

```bash
proto/bin/protoc.exe object_detection/protos/*.proto --python_out=. 
```

### Finally set environment variables, so you can import object_detection folder to your python script:

- <PATH_TO_TF>/models/research
- <PATH_TO_TF>/models/research/slim

### To check out if everything is working correctly, import the following module:

```python
python
>> import object_detection
>> 
```
