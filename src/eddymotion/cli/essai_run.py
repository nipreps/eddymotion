

import os
import sys
import numpy as np

from parser import parse_args
from eddymotion.data.dmri import DWI
from eddymotion.estimator import EddyMotionEstimator, _prepare_registration_data
from eddymotion.data.splitting import lovo_split
from eddymotion.model import ModelFactory
from pathlib import Path

# Open the data with the given file path
dwi_dataset = DWI.from_filename('/home/esavary/tests-eddymotion/dwi.h5')
print(len(dwi_dataset))

data_train, data_test  = lovo_split(dwi_dataset, 38, with_b0=True)

#print(np.shape(train_data),np.shape(test_data), np.shape(train_gradients), np.shape(test_gradients))

#debug fit
dwmodel = ModelFactory.init(model="avg")
dwmodel.fit(data_train[0])

predicted = dwmodel.predict(data_test[1])
print(np.shape(predicted),np.shape(data_test[0]))
print(dwi_dataset.affine)
fixed, moving = _prepare_registration_data(data_test[0], predicted, dwi_dataset.affine, 38, Path("/home/esavary/test_temp/"), "b0")
print(fixed)

# Initialize the EddyMotionEstimator
estimator = EddyMotionEstimator()
estimated_affines = EddyMotionEstimator.fit(dwi_dataset, models=["avg"])
print(estimated_affines)