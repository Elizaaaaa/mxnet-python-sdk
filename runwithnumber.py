import mxnet as mx
import tarfile
import numpy as np
import time
import json

from sagemaker.session import Session
from sagemaker.mxnet import MXNetModel
from mxnet.gluon.data.vision import transforms

mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx')

with tarfile.open('onnx_model.tar.gz', mode='w:gz') as archive:
    archive.add('resnet50v2.onnx')

model_data = Session().upload_data(path='onnx_model.tar.gz', key_prefix='model')
role = 'arn:aws:iam::841569659894:role/sagemaker-access-role'

mxnet_model = MXNetModel(model_data=model_data,
                         entry_point='resnet50.py',
                         role=role,
                         image='763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.4.1-gpu-py36-cu100-ubuntu16.04',
                         py_version='py3',
                         framework_version='1.4.1')

predictor = mxnet_model.deploy(initial_instance_count=1, instance_type='ml.p3.8xlarge')

def do_pred():
    data = np.random.rand(1, 3, 224, 224)
    start_time = time.time()
    scores = predictor.predict(data)
    end_time = time.time()
    
    return end_time-start_time

costtime = do_pred()
print("this run cost {}s".format(costtime))
