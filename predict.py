import mxnet as mx
import numpy as np
import time
import json

from sagemaker.session import Session
from sagemaker.mxnet import MXNetPredictor

#predictor = MXNetPredictor('mxnet-inference-2019-09-11-23-33-38-737', Session())
predictor = MXNetPredictor('mxnet-inference-2019-09-13-18-42-06-926', Session())

data = np.random.rand(1,3,224,224)
input_data = {"instances":data}
start = time.time()
scores = predictor.predict(data)
end = time.time()

print(end-start)
