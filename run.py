import mxnet as mx
import tarfile
import numpy as np
import time

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
                         py_version='py3',
                         framework_version='1.4.1')

predictor = mxnet_model.deploy(initial_instance_count=1, instance_type='ml.p3.8xlarge')

img_path = mx.test_utils.download('https://s3.amazonaws.com/onnx-mxnet/examples/mallard_duck.jpg')
img = mx.image.imread(img_path)

def preprocess(img):
    transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img

input_image = preprocess(img)
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/synset.txt')

def do_pred():
    start_time = time.time()
    scores = predictor.predict(input_image.asnumpy())
    end_time = time.time()
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    a = np.argsort(scores)[::-1]

    for i in a[0:5]:
        print('class=%s ; probability=%f' %(labels[i],scores[i]))

    return end_time-start_time

costtime = do_pred()
print("this run cost {}s".format(costtime))