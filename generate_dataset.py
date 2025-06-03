import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.integrate import quad, cumtrapz

import matplotlib.pyplot as plt

import librosa

class GenerateSignal():
  def __init__(self):
    pass
  
  @staticmethod
  def _calc_speed(rpm_line):
    return rpm_line * 2 * np.pi * 0.49
  
  @staticmethod
  def _calc_distance(speed_line):
    # Наклон можно понять по изменению скорости, т.е. по производной speed_line (SL). SL == 0 -> едем прямо. SL < 0 - в горку SL > 0 - с горки
    # Тогда путь будет расстояние GPS = V * cos(V')
    dvdt = savgol_filter(speed_line, 5, 3, deriv=1, delta=1)
    if np.max(np.abs(dvdt)) > 1:
       dvdt /= np.max(np.abs(dvdt))
    return cumtrapz(speed_line * np.cos(np.abs(dvdt)), initial=0)
    
  @staticmethod
  def _clean_1(wheel_rpm, speed, distance, datalen):
    # Данные просто заполняются без строгих зависимостей
    return pd.date_range(start='2025-05-31', periods=datalen, freq='S'), wheel_rpm * np.ones((datalen)), speed * np.ones((datalen)), np.linspace(0, distance, datalen)
  
  @staticmethod
  def _clean_2(wheel_rpm, datalen):
    # Данные моделируются. Исходя из доступных датчиков - можно задать только 1 параметр для моделирования
    # Пусть выбранный параметр является wheel_rpm до накладывания аномалий
    # Тогда скорость задается через speed = wheel_rpm * wheel_len = rpm * 2 * np.pi * R
    
    # Данный набор данных является сильно коррелирующими и снимается машины
    rpm_line = wheel_rpm * np.ones((datalen))
    speed_line = GenerateSignal._calc_speed(rpm_line)
    
    # Данные GPS слабее коррелируют с данными скорости, поскольку снимают проекцию передвижения, не учитывая набор высоты
    # Будем считать, что у автомобиля постоянная мощность достаточная для поддержаня движения, поэтому в горку едет медленней а с горки быстрее. По прямой постоянно
    # Данные пройденного пути являются интегралом от скорости с учетом наклона поверхности

    distance_line = GenerateSignal._calc_distance(speed_line)
    return pd.date_range(start='2025-05-31', periods=datalen, freq='S'), rpm_line, speed_line, distance_line
  
  @staticmethod
  def add_hills(time_line, rpm_line, speed_line, distance_line, freq, ampl):
    """Симуляция движения по ухабам \n freq - Частота, dur - Амплитуда"""
    # print(f'Применение эффекта катания по ухабам с частотой {freq} и длительностью {dur}')
    # freq - частота ухабов
    # dur - амплитуда ухабов

    sin_part = ampl * np.sin(freq * 2 * np.pi * np.linspace(0, len(time_line), len(time_line)))
    n_rpm_line = sin_part + rpm_line
    n_seed_line = GenerateSignal._calc_speed(n_rpm_line)
    return time_line, n_rpm_line, n_seed_line, GenerateSignal._calc_distance(n_seed_line)

class ApplyAnomaly():
  def __init__(self):
    pass
  
  @staticmethod
  def wheel_slip(time_line, rpm_line, speed_line, distance_line, freq, dur):
    """Проскальзывание колеса"""
    # print(f'Применение проскальзывания с частотой {freq} и длительностью {dur}')
    # freq - сколько проскальзываний в час
    # dur - длительность проскальзывания
    slip_ampl = 1.4 # амплитуда проскальзывания колеса, считается постоянной
    step = 60 * 60 // freq # перевод штук проскальзываний в час в интервал между началами
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step): # будем менять данные с заданным интервалом
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      # Дальше хочу чтобы проскальзывание было параболическим изменением rpm, сначала теряем зацем, а потом его находим и возвращаемся
      x_points = np.array([i, i + dur / 2, i + dur], dtype=np.int32) + random_shift # точки для поиска параболы
      y_points = [rpm_line[x_points[0]], slip_ampl * (rpm_line[x_points[0]] + rpm_line[x_points[2]]) / 2, rpm_line[x_points[2]]]
      A = np.vstack([np.square(x_points), x_points, np.ones(3)]).T
      a, b, c = np.linalg.solve(A, y_points) # находим коэффициенты параболы
      xline = np.arange(x_points[0], x_points[2], 1) 
      rpm_line[x_points[0]:x_points[2]] = a * xline**2 + b * xline + c # восстанавливаем данные с аномалией
    return time_line, rpm_line, speed_line, distance_line
  
  @staticmethod
  def gps_loss(time_line, rpm_line, speed_line, distance_line, freq, dur):
    """Потеря сигнала GPS"""
    # print(f'Применение потери GPS с частотой {freq} и длительностью {dur}')
    # freq - сколько потерей сигнала в час
    # dur - длительность потери
    step = 60 * 60 // freq
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step):
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      xline = np.arange(i, i + dur, 1, dtype=np.int32) + random_shift
      distance_line[xline] = [None] * len(xline)
    return time_line, rpm_line, speed_line, distance_line
  
  @staticmethod
  def sampling_shift(time_line, rpm_line, speed_line, distance_line, freq, dur):
    """Дрейф значений скорости (внезапно фиксится)"""
    step = 60 * 60 // freq
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step):
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      xline = np.arange(i, i + dur, 1, dtype=np.int32) + random_shift
      speed_shift = np.linspace(0, 1, len(xline))
      speed_line[xline] += speed_shift

    return time_line, rpm_line, speed_line, distance_line
  
  @staticmethod
  def white_noise(time_line, rpm_line, speed_line, distance_line, freq, dur):
    """Добавление белого шума"""
    step = 60 * 60 // freq
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step):
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      xline = np.arange(i, i + dur, 1, dtype=np.int32) + random_shift
      rpm_line[xline] += np.random.normal(0, 0.2, len(xline))
      
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step):
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      xline = np.arange(i, i + dur, 1, dtype=np.int32) + random_shift
      speed_line[xline] += np.random.normal(0, 0.2, len(xline))
    
    for i in range(step + dur + 1, len(rpm_line) - step - dur, step):
      random_shift = np.random.randint(-step // 3, step // 3) # чтобы убрать периодичность, добавим случайный сдвиг начала
      xline = np.arange(i, i + dur, 1, dtype=np.int32) + random_shift
      distance_line[xline] += np.random.normal(0, 0.2, len(xline))

    return time_line, rpm_line, speed_line, distance_line

def generate(wheel_rpm, speed, distance, datalen, anomalies, hills, freq, ampl):
  """
  anomalies = [[method_1_name : str, frequency : float, duration : float],
               [method_2_name : str, frequency : float, duration : float],
               ...]
  """
  # time_line, rpm_line, speed_line, distance_line = GenerateSignal._clean_1(wheel_rpm, speed, distance, datalen)
  time_line, rpm_line, speed_line, distance_line = GenerateSignal._clean_2(wheel_rpm, datalen)
  if hills:
    time_line, rpm_line, speed_line, distance_line = GenerateSignal.add_hills(time_line, rpm_line, speed_line, distance_line, freq, ampl)
  for element in anomalies:
    time_line, rpm_line, speed_line, distance_line = getattr(ApplyAnomaly, element[0])(time_line, rpm_line, speed_line, distance_line, element[1], element[2])
  return time_line, rpm_line, speed_line, distance_line
