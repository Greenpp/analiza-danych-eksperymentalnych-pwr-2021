# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

DATA_FILE = './data_final/plastic.dat'
PLOT_DIR = './lab4_plots'


def load_data() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f'File not found at {data_file.absolute()}')

    return pd.read_csv(data_file, header=None, comment='@')

def make_plots(data):
    for i in range(data.shape[1]):
        sns.scatterplot(data=data.iloc[:,i])        
        plt.savefig(PLOT_DIR + '/' + "Scatter_" + str(i))
        plt.clf()        
        
        sns.histplot(data=data.iloc[:,i])
        plt.title(f'X{i}')
        plt.xlabel('Średnia')
        plt.ylabel('Liczność')
        plt.savefig(PLOT_DIR + '/' + "Hist_" + str(i))
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

# %%
data = load_data()
# %%
basic_analysis(data)
# %%
make_plots(data)
# %%
