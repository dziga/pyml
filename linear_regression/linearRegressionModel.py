from progress_bar import printProgressBar

class LinearRegression:
  def __init__(self, fix, epocs):
    self.fix = fix
    self.t0 = 1
    self.t1 = 1
    self.epocs = epocs  

  def fit(self, x, y):
    _t0 = self.t0 or 1
    _t1 = self.t1 or 1
    _epocs = self.epocs or 100
    _m = len(x)

    printProgressBar(0, _epocs, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range(0, _epocs):
      printProgressBar(i, _epocs - 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
      _sum0 = 0
      _sum1 = 0
      for j in range(1, _m):
        _sum0 += _t0 + _t1*x[j] - y[j]
      for j in range(1, _m):
        _sum1 += (_t0 + _t1*x[j] - y[j])*x[j]
      
      _t0 = _t0 - self.fix * _sum0/_m
      _t1 = _t1 - self.fix * _sum1/_m
    self.t0 = _t0 
    self.t1 = _t1

  def predict(self, x):
    return [(self.t0 + self.t1*x[i])[0] for i in range(len(x))]

  