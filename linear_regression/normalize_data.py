from statistics import mean, median

def normalize(x, coeficient):
  return [i * coeficient for i in x]