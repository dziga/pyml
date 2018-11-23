from sklearn import datasets
import numpy as np

def getDiabetesData(featuresSlice):
  diabetes = datasets.load_diabetes()
  data_chunk = 442
  data = diabetes.data[:data_chunk, np.newaxis, featuresSlice]
  target = diabetes.target[:data_chunk]
  return data, target

def getDiabetesDataMultipleFeatures(featuresSlice):
  diabetes = datasets.load_diabetes()
  data_chunk = 442
  data = diabetes.data[:data_chunk, :featuresSlice]
  target = diabetes.target[:data_chunk]
  return data, target

def getHousingData():
  diabetes = datasets.load_boston()
  data_chunk = 506
  column = 5
  data = diabetes.data[:data_chunk, column]
  target = diabetes.target[:data_chunk]
  return data, target