# %%
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import moment

DATA_FILE = './data_final/plastic.dat'
PLOT_DIR = './lab4_plots'


def load_data() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f'File not found at {data_file.absolute()}')

    return pd.read_csv(data_file, header=None, comment='@')

def make_plots(data):
    sns.histplot(data=data.iloc[:, 0], stat='count')
    plt.title(f'Strength')
    plt.xlabel('Średnia')
    plt.ylabel('Liczność')
    plt.savefig(PLOT_DIR + '/' + "Hist_Str")
    plt.clf()

    sns.histplot(data=data.iloc[:, 1], stat='count')
    plt.title(f'Temperature')
    plt.xlabel('Średnia')
    plt.ylabel('Liczność')
    plt.savefig(PLOT_DIR + '/' + "Hist_Temp")
    plt.clf()

    sns.histplot(data=data.iloc[:, 2], stat='count')
    plt.title(f'Pressure')
    plt.xlabel('Średnia')
    plt.ylabel('Liczność')
    plt.savefig(PLOT_DIR + '/' + "Hist_Press")
    plt.clf()

def make_scatterplots(data):
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1])  
    plt.xlabel('Strength')
    plt.ylabel('Temperature')
    plt.savefig(PLOT_DIR + '/' + "Scatter_Str_Temp")
    plt.clf()

    sns.scatterplot(x=data.iloc[:, 1], y=data.iloc[:, 2])
    plt.xlabel('Temperature')
    plt.ylabel('Pressure')
    plt.savefig(PLOT_DIR + '/' + "Scatter_Temp_Press")
    plt.clf()

    sns.scatterplot(x=data.iloc[:, 2], y=data.iloc[:, 0])
    plt.xlabel('Pressure')
    plt.ylabel('Strength')
    plt.savefig(PLOT_DIR + '/' + "Scatter_Press_Str")
    plt.clf()

def basic_analysis(data):
    desc = data.describe()
    print(f'Liczność:\n{desc.loc["count"]}')
    print(f'Średnia wartość:\n{desc.loc["mean"]}')
    print(f'Odchylenie standardowe::\n{desc.loc["std"]}')
    print(f'Najmniejsza wartość:\n{desc.loc["min"]}')
    print(f'Największa wartość:\n{desc.loc["max"]}')
    print(f'1 kwartyl:\n{desc.loc["25%"]}')
    print(f'Mediana:\n{desc.loc["50%"]}')
    print(f'3 kwartyl:\n{desc.loc["75%"]}')

def advanced_analysis(data):
    atributes = {'Strength': 0, 'Temperature': 1, 'Pressure': 2}
    for key, value in atributes.items():
        print(f'{key}:')
        moments = []
        for i in range(4):
            moments.append(moment(data.iloc[:, value], moment=i))
        for m in range(len(moments)):
            print(f'Moment {m+1}: {moments[m]}')
        mean = np.mean(data.iloc[:, value])
        print(f'Średnia: {mean}')
        skew = moments[2] / moments[1] ** (3 / 2)
        print(f'Asymetria: {skew}')
        focus = moments[3] / moments[1] ** 2
        print(f'Skupienie: {focus}')
        var = moments[1] ** 0.5 / mean * 100
        print(f'Zmienność: {var}%')
# %%
data = load_data()
# %%
basic_analysis(data)
# %%
advanced_analysis(data)
# %%
make_plots(data)
# %%
make_scatterplots(data)
# %%