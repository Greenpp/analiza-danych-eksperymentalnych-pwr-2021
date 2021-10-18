from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

DATA_PATH = Path('data')
RESULTS_PATH = Path('results')
PLOTS_PATH = Path('plots')

for i in tqdm(range(4)):
    data_file = f'data{i + 1}'
    data_file_path = DATA_PATH / data_file

    res_path = RESULTS_PATH / f'res_{i + 1}.txt'
    with open(res_path, 'w') as f:
        f.write(f'FILE: {data_file}\n\n')

        data = pd.read_csv(data_file_path)
        f.write('Wartości opisowe:\n')
        f.write('\nŚrednia:\n')
        f.write(str(data.mean().round(2)))
        f.write('\nWariacja:\n')
        f.write(str(data.var().round(2)))
        f.write('\nMediana:\n')
        f.write(str(data.median().round(2)))
        f.write('\nKorelacja:\n')
        f.write(str(data.corr().round(2)))
        f.write('\n' * 3)
        x = data['x'].to_numpy().reshape((-1, 1))
        y = data['y'].to_numpy()
        reg = LinearRegression().fit(x, y)

        f.write('\nRegresja:\n')
        b1 = round(reg.coef_[0], 5)
        b0 = round(reg.intercept_, 5)
        f.write(f'Coeff B0:{b0} B1:{b1}')
    sns.regplot(data=data, x='x', y='y', ci=0)
    plt.title(f'Seria {i + 1}')

    plot_path = PLOTS_PATH / f'plot_{i + 1}'
    plt.savefig(plot_path)
    plt.clf()