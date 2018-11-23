import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from linearRegressionModel import LinearRegression
from data import getDiabetesData, getHousingData
from sklearn.externals import joblib
import os
from normalize_data import normalize

limiter = -20
featuresSlice = 2

#load data
data, target = getDiabetesData(featuresSlice)

#normalize data
data = data * 10
target = target / 100

#features data split
x_train = data[:limiter]
x_test = data[limiter:]

#target data split
y_target_train = target[:limiter]
y_target_test = target[limiter:]

lr = LinearRegression(0.001, 1000)

#train
lr.fit(x_train, y_target_train)

# load model if already trained
# lr = joblib.load(os.path.join('./scikit', 'diabetes-model.joblib'))

#predict
res = lr.predict(x_test)

#de-normalize data
x_test = x_test / 10
y_target_test = y_target_test * 100
res = [i * 100 for i in res]

plt.scatter(x_test, y_target_test,  color='black')
plt.plot(x_test, res, color='blue', linewidth=3)

plt.show()

print("Mean squared error: %.2f" % mean_squared_error(y_target_test, res))
print('Variance score: %.2f' % r2_score(y_target_test, res))

# save model
# joblib.dump(lr, os.path.join('./scikit', 'housing-model.joblib'))