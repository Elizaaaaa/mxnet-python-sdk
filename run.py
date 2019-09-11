import mxnet as mx
import tarfile

from sagemaker.session import Session
from sagemaker.mxnet import MXNetModel

mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx')

with tarfile.open('onnx_model.tar.gz', mode='w:gz') as archive:
    archive.add('resnet50v2.onnx')

model_data = Session().upload_data(path='onnx_model.tar.gz', key_prefix='model')
role = 'arn:aws:iam::841569659894:role/sagemaker-access-role'

mxnet_model = MXNetModel(model_data=model_data,
                         entry_point='resnet152.py',
                         role=role,
                         py_version='py3',
                         framework_version='1.4.1')

predictor = mxnet_model.deploy(initial_instance_count=1, instance_type='ml.p3.8xlarge')