import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from linearRegressionModelMultiple import LinearRegressionMultiple
from data import getDiabetesDataMultipleFeatures, getHousingData
from sklearn.externals import joblib
import os
from normalize_data import normalize

limiter = -20
featuresSlice = 3

#load data
data, target = getDiabetesDataMultipleFeatures(featuresSlice)

#normalize data
data = data * 10
target = target / 100

# add vector of 1 to data
v = np.ones((442, 1))
data = np.c_[v, data]

x_train = data[:limiter]
x_test = data[limiter:]

y_target_train = target[:limiter]
y_target_test = target[limiter:]

# lr = LinearRegressionMultiple(0.001, 10000)
# lr.fit(x_train, y_target_train)

lr = joblib.load(os.path.join('./scikit', 'diabetes-model-multiple.joblib'))

res = lr.predict(x_test)

print(x_test)
print(y_target_test)

plt.scatter(x_test[:, 1], y_target_test,  color='black')
plt.plot(x_test[:, 1], res, color='blue', linewidth=3)

plt.show()

x_test = x_test / 10
y_target_test = y_target_test * 100
res = [i * 100 for i in res]

print("Mean squared error: %.2f" % mean_squared_error(y_target_test, res))
print('Variance score: %.2f' % r2_score(y_target_test, res))

# joblib.dump(lr, os.path.join('./scikit', 'diabetes-model-multiple.joblib'))