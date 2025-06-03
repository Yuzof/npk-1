import numpy as np
import pandas as pd
import datetime

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,  Matern, ExpSineSquared, RBF
from matrixprofile import *
from scipy.signal import medfilt

import matplotlib.pyplot as plt

class DetectAnomaly():
  def __init__(self):
    pass
  
  # @staticmethod
  # def wheel_slip(time_line, rpm_line, speed_line, distance_line):
  #   """Детектирование проскальзывания колеса"""
  #   # print(f'Детектирование проскальзывания с частотой {freq} и длительностью {dur}')
  #   return time_line, rpm_line, speed_line, distance_line

  # @staticmethod
  # def gps_loss(time_line, rpm_line, speed_line, distance_line):
  #   """Детектирование потери сигнала GPS"""
  #   # print(f'Детектирование потери GPS с частотой {freq} и длительностью {dur}')
  #   return time_line, rpm_line, speed_line, distance_line
  
  @staticmethod
  def GPR(tline, yline):
    """Фильтрация с изпользованием GPR"""
    # создадим маску, чтобы убрать плозие значения
    converted_tline = np.array([datetime.datetime.strptime(str(element), "%Y-%m-%d %H:%M:%S").timestamp() for element in tline])
    converted_tline -= converted_tline[0]
    yline = np.array(yline)
    # print(converted_tline)
    mask = [i for i in range(len(tline)) if not np.isnan(yline[i])]
    print('Calculating GPR...')
    kernel = Matern()
    gpr = GaussianProcessRegressor(
            kernel = kernel,
            alpha = 1e-8,
            n_restarts_optimizer = 0,
            random_state = 42
        ).fit(converted_tline[mask].reshape(-1, 1), yline[mask].reshape(-1, 1))
    print('Finished with score : ', gpr.score(converted_tline[mask].reshape(-1, 1), yline[mask].reshape(-1, 1)))
    print('Predicting ...')
    y_mean = gpr.predict(converted_tline.reshape(-1, 1))
    return y_mean
  
  @staticmethod
  def MatrixProfile(tline, yline):
    """Матричное профилирование"""
    prof_len = 64
    mp = matrixProfile.stomp(np.array(yline), prof_len)
    mp_adj = np.append(mp[0],np.zeros(prof_len-1)+np.nan)

    # filtered = medfilt(mp_adj, kernel_size=51)
    # Bonus: calculate the corrected arc curve (CAC) to do semantic segmantation.
    # cac = fluss.fluss(mp[1], m)

    return mp_adj


def detect(time_line, rpm_line, speed_line, distance_line, detectors):
  """
  detectors = [[method_1_name : str, rpm : bool, speed : bool, distance : bool],
              [method_2_name : str, rpm : bool, speed : bool, distance : bool]
              ...]
  """
  for element in detectors:
    if element[1]:
      rpm_line = getattr(DetectAnomaly, element[0])(time_line, rpm_line)
    if element[2]:
      speed_line = getattr(DetectAnomaly, element[0])(time_line, speed_line)
    if element[3]:
      distance_line = getattr(DetectAnomaly, element[0])(time_line, distance_line)
  return pd.DataFrame({'time' : time_line, 'x' : rpm_line, 'y' : speed_line, 'z' : distance_line})

def find_intervals(yline):
  meanvalue = np.mean(yline)
  indices = np.where(yline > 2 * meanvalue)[0]
  intervals = []
  if len(indices) > 0:
    intervals.append(indices[0])
    for i in range(0, len(indices) - 1, 1):
      if indices[i+1] - indices[i] > 1:
        intervals.append(indices[i])
        intervals.append(indices[i+1])
    intervals.append(indices[-1])
  return intervals

def get_intervals(time_line, rpm_line, speed_line, distance_line):
  ans = [[*find_intervals(rpm_line)], [*find_intervals(speed_line)]]
  return [[time_line[ans[0]]],
          [time_line[ans[1]]]]

if __name__ == '__main__':
  import matplotlib.pyplot as plt

  a = np.linspace(0, 2*np.pi * 10, 2000)
  b = np.sin(a)

  b[1000:1050] += np.sin(b[1000:1050])

  mp = DetectAnomaly.MatrixProfile(a, b)
  mp_adj = np.append(mp[0],np.zeros(32-1)+np.nan)

  plt.plot(b)
  plt.plot(mp_adj)
  plt.show()