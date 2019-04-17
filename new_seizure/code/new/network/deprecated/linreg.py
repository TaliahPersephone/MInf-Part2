import numpy as np
from sklearn.linear_model import LinearRegression
from histdata import load_data
import logging

seed = 4760203



print('Loading data, seed = {}'.format(seed))
(x_train,y_train),(x_test,y_test) = load_data(seed=seed)

print('Fitting linear regression model')

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

TP = np.sum(np.logical_and(y_pred==1,y_test==1))
FP = np.sum(np.logical_and(y_pred==1,y_test==0))
TN = np.sum(np.logical_and(y_pred==0,y_test==0))
FN = np.sum(np.logical_and(y_pred==0,y_test==1))

print('TP={}\tTN={}\nFP={}\tFN={}\n'.format(TP,TN,FP,FN))


print('TPR={}\tSPC={}'.format(TP/(TP+FN),TN/(FP+TN)))

l = logging.getLogger('logs/linreg.log')

l.info('seed = {}\n\n'.format(seed))

l.info('TP={}\tTN={}\nFP={}\tFN={}\n'.format(TP,TN,FP,FN))

l.info('TPR={}\tSPC={}'.format(TP/(TP+FN),TN/(FP+TN)))

