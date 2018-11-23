import numpy as np
from progress_bar import printProgressBar

class LinearRegressionMultiple:
  def __init__(self, fix, epocs):
    self.fix = fix
    self.epocs = epocs  
    self.tetha = [1,1,1,1]

  def fit(self, x, y):
    _epocs = self.epocs
    _m, _n = x.shape

    printProgressBar(0, _epocs - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for e in range(0, _epocs):
      printProgressBar(e, _epocs - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
      
      for j in range(_n):  
        _sum = 0
        for i in range(0, _m):
          _sum += (np.dot(self.tetha, x[i]) - y[i]) * x[i][j]

        self.tetha[j] = self.tetha[j] - self.fix * _sum/_m 

  def predict(self, x):
    return [np.dot(self.tetha, x[i]) for i in range(len(x))]