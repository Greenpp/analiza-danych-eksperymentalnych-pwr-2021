# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = './data2/d.csv'
CLASSES = np.linspace(10, 70, 7)


def load_data() -> pd.DataFrame:
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        raise FileNotFoundError(f'File not found at {data_file.absolute()}')

    data = pd.read_csv(data_file)
    return data


def make_hist(data: pd.Series):
    sns.barplot(x=CLASSES, y=data)
    plt.title(f'Seria {data.name}')
    plt.xlabel('Średnia')
    plt.ylabel('Osobniki')
    plt.show()
    plt.clf()


def get_mean(data):
    mean = (CLASSES * data).sum() / data.sum()

    return mean


def get_moment(data, mean, k):
    moment = ((CLASSES - mean) ** k * data).sum() / data.sum()

    return moment


# %%
data = load_data()

for col in data:
    make_hist(data[col])
    mean = get_mean(data[col])
    print(f'Średnia: {mean}')
    moments = [get_moment(data[col], mean, i + 1) for i in range(4)]
    for i, m in enumerate(moments, 1):
        print(f'Moment {i}: {m}')

    skew = moments[2] / moments[1] ** (3 / 2)
    print(f'Asymetria: {skew}')

    focus = moments[3] / moments[1] ** 2
    print(f'Skupienie: {focus}')

    var = moments[1] ** 0.5 / mean * 100
    print(f'Zmienność: {var}%')

# %%
