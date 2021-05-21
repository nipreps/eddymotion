# NumPy is a fundamental Python library for working with arrays
import numpy as np
# import the class from the library
from eddymotion.dmri import DWI
from eddymotion.model import AverageDWModel, ModelFactory
from eddymotion.viz import plot_dwi

# create a new DWI object, with only gradient information that is random
dmri_dataset = DWI.from_filename("./data/dwi.h5")

model = AverageDWModel(
    dmri_dataset.gradients,
    S0=dmri_dataset.bzero,
    th_low=100,
    th_high=1000,
    bias = True,
    stat = 'mean'
)

data_train, data_test = dmri_dataset.logo_split(10)
model.fit(data_train[0])
predicted = model.predict(data_test[1])

plot_dwi(predicted, dmri_dataset.affine, gradient=data_test[1])
plot_dwi(data_test[0], dmri_dataset.affine, gradient=data_test[1])

#Test if the two different ways of initializing the model 
#gives the same model
model2 = ModelFactory.init(
    gtab=data_train[1],
    model="avg",
    S0=dmri_dataset.bzero,
    th_low=100,
    th_high=1000,
    bias = True,
    stat = 'mean'
)
model2.fit(data_train[0])
predicted2 = model2.predict(data_test[1])

predicted1 == predicted 2