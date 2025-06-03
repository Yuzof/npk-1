from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os

from generate_dataset import generate, ApplyAnomaly, GenerateSignal
from anomaly_detector import detect, get_intervals, DetectAnomaly

def get_features_list():
  methods = [method for method in dir(GenerateSignal) if not method.startswith('_')]
  return [[method, getattr(GenerateSignal, method).__doc__] for method in methods]

def create_checks_for_features(item):
  return ui.TagList(ui.input_checkbox(item[0] + '_feature_check', item[1], False))

def get_anomalies_list():
  methods = [method for method in dir(ApplyAnomaly) if not method.startswith('__')]
  return [[method, getattr(ApplyAnomaly, method).__doc__] for method in methods]

def create_checks_for_anomaly(item):
  return ui.TagList(ui.input_checkbox(item[0] + '_check', item[1], False),
                    ui.layout_columns(
                      ui.input_numeric(item[0] + 'freq', 'Частота шт/час', value=10),
                      ui.input_numeric(item[0] + 'dur', 'Длительность сек', value=10)
                    )
                  )

def get_anomalies_detectors_lsit():
  methods = [method for method in dir(DetectAnomaly) if not method.startswith('__')]
  return [[method, getattr(DetectAnomaly, method).__doc__] for method in methods]

def create_checks_for_detectors(item):
  return ui.TagList(ui.input_checkbox(item[0] + '_detector_check', item[1], False),
                    ui.layout_columns(
                        ui.input_checkbox(item[0] + '_detector_rpm_check', 'Wheel RPM', False),
                        ui.input_checkbox(item[0] + '_detector_speed_check', 'Speed', False),
                        ui.input_checkbox(item[0] + '_detector_distance_check', 'Distance', False),
                    )
                )
  
app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel('Генерация',
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_numeric('wheel_rpm', 'wheel_rpm', value=10.0),
                    ui.layout_columns(
                        ui.input_checkbox('hills', 'Hills', False),
                        ui.input_numeric('hills_freq', 'Частота', value=10.0),
                        ui.input_numeric('hills_ampl', 'Амлитуда', value=10.0)
                    ),
                    #ui.input_numeric('speed', 'speed', value=20.0),
                    #ui.input_numeric('distance', 'distance', value=100.0),
                    ui.input_numeric('time_dur_min', 'Время моделирования сек', value=3600),
                    *[create_checks_for_anomaly(item) for item in get_anomalies_list()],
                    ui.download_button('download_data', 'Сохранить данные'),
                    width=3),
                ui.panel_main(
                    ui.layout_columns(
                      ui.output_plot('generic_plot_1', fill=False),
                      ui.output_plot('generic_plot_2', fill=False)
                    )
                ),
            ),
        ),
        ui.nav_panel('Анализ',
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_file('upload', 'Загрузите CSV-файл', accept=['.csv']),
                    *[create_checks_for_detectors(item) for item in get_anomalies_detectors_lsit()],
                    ui.download_button('download_intervals', 'Сохранить интервалы'),
                    width=3,
                ),
                ui.panel_main(
                      ui.output_plot('input_plot', fill=False),
                      ui.output_plot('detected_plot', fill=False),
                ),
            ),
        ),
    ),
)

def server(input, output, session):
    # Хранилище генерируемых данных
    plot_data = reactive.Value(pd.DataFrame({'time' : [], 'x': [], 'y': [], 'z' : []}))
    
    # Хранилище загруженных данных
    uploaded_df = reactive.Value(pd.DataFrame({'time' : [], 'x': [], 'y': [], 'z' : []}))
    
    # Хранилище после обработки
    detected_df = reactive.Value(pd.DataFrame({'time' : [], 'x': [], 'y': [], 'z' : []}))
    
    # Генерация
    @reactive.Effect
    def update_plot_data():
        # time, x, y, z = generate(input.wheel_rpm(), input.speed(), input.distance(), input.time_dur_min(), gather_aniomalies())
        time, x, y, z = generate(input.wheel_rpm(), 0, 0, input.time_dur_min(), gather_aniomalies(), input.hills(), input.hills_freq(), input.hills_ampl())
        plot_data.set(pd.DataFrame({'time' : time, 'x': x, 'y': y, 'z' : z}))
    
    @reactive.calc
    def gather_aniomalies():
      """        
      anomalies = [[method_1_name : str, frequency : float, duration : float],
                  [method_2_name : str, frequency : float, duration : float],
                  ...]
      """
      return [[element[0], getattr(input, element[0] + 'freq').get(), getattr(input, element[0] + 'dur').get()] 
              for element in get_anomalies_list() 
                if getattr(input, element[0] + '_check').get()]
    
    @reactive.calc
    def gather_detectors():
        """
        detectors = [[method_1_name : str, rpm : bool, speed : bool, distance : bool],
                    [method_2_name : str, rpm : bool, speed : bool, distance : bool]
                    ...]
        """
        return [[element[0], 
                 getattr(input, element[0] + '_detector_rpm_check').get(),
                 getattr(input, element[0] + '_detector_speed_check').get(),
                 getattr(input, element[0] + '_detector_distance_check').get()]
                for element in get_anomalies_detectors_lsit() if getattr(input, element[0] + '_detector_check').get()]

    # График перед генерацией
    @output
    @render.plot
    def generic_plot_1():
        df = plot_data.get()
        fig, ax = plt.subplots()
        ax.plot(df['time'], df['x'], linewidth=2, label = 'wheel_rpm')
        ax.plot(df['time'], df['y'], linewidth=2, label = 'speed')
        ax.set_title('Предварительные графики')
        ax.set_xlabel('Время')
        ax.set_ylabel('rpm, speed, distance')
        ax.grid(True)
        ax.legend(loc = 'upper left')
        return fig
    
        # График перед генерацией
    @output
    @render.plot
    def generic_plot_2():
        df = plot_data.get()
        fig, ax = plt.subplots()
        ax.plot(df['time'], df['z'], linewidth=2, label = 'distance')
        ax.set_title('Предварительные графики')
        ax.set_xlabel('Время')
        ax.set_ylabel('rpm, speed, distance')
        ax.grid(True)
        ax.legend(loc = 'upper left')
        return fig

    @render.download()
    def download_data():
        filename = f'data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        plot_data.get().to_csv(filename, index=False)
        path = os.path.join(os.path.dirname(__file__), filename)
        return path
    
    @render.download()
    def download_intervals():
        filename = f'intervals_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df = detected_df.get()
        pd.DataFrame(get_intervals(df['time'], df['x'], df['y'], df['z'])).to_csv(filename, index = False, header = None)
        path = os.path.join(os.path.dirname(__file__), filename)
        return path

    # Обработка загруженных данных
    @reactive.Effect
    def handle_upload():
        file = input.upload()
        if not file:
            uploaded_df.set(None)
            return

        uploaded_df.set(pd.read_csv(file[0]['datapath']))
        
    @output
    @render.plot
    def input_plot():
        df = uploaded_df.get()
        if df is None:
            return None

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(df['time'], df['x'], linewidth=2, label = 'wheel_rpm')
        ax[0].plot(df['time'], df['y'], linewidth=2, label = 'speed')
        ax[0].set_title('Графики после детектирования')
        ax[0].set_xlabel('Время')
        ax[0].set_ylabel('rpm, speed')
        ax[0].grid(True)
        ax[0].legend(loc = 'upper left')
        ax[0].set_xticks([df['time'][num] for num in range(0, len(df['time']), len(df['time']) // 10)])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
        
        ax[1].plot(df['time'], df['z'], linewidth=2, label = 'distance')
        ax[1].set_title('Графики после детектирования')
        ax[1].set_xlabel('Время')
        ax[1].set_ylabel('distance')
        ax[1].grid(True)
        ax[1].legend(loc = 'upper left')
        ax[1].set_xticks([df['time'][num] for num in range(0, len(df['time']), len(df['time']) // 10)])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
        return fig

    @output
    @render.plot
    def detected_plot():
        df = uploaded_df.get()
        if df is None:
            return None
        
        df = detect(df['time'], df['x'], df['y'], df['z'], gather_detectors())
        detected_df.set(df)

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(df['time'], df['x'], linewidth=2, label = 'wheel_rpm')
        ax[0].plot(df['time'], df['y'] - np.max(df['y']), linewidth=2, label = 'speed')
        ax[0].set_title('Графики после детектирования')
        ax[0].set_xlabel('Время')
        ax[0].set_ylabel('rpm, speed')
        ax[0].grid(True)
        ax[0].legend(loc = 'upper left')
        ax[0].set_xticks([df['time'][num] for num in range(0, len(df['time']), len(df['time']) // 10)])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
        
        ax[1].plot(df['time'], df['z'], linewidth=2, label = 'distance')
        ax[1].set_title('Графики после детектирования')
        ax[1].set_xlabel('Время')
        ax[1].set_ylabel('distance')
        ax[1].grid(True)
        ax[1].legend(loc = 'upper left')
        ax[1].set_xticks([df['time'][num] for num in range(0, len(df['time']), len(df['time']) // 10)])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
        return fig

app = App(app_ui, server)