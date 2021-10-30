# %%
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

DATA_DIR = Path('./data')
EXPORT_DIR = Path('./export_lab2')

FILES = [
    'data1.csv',
    'data2.csv',
    'data3.csv',
    'data4.csv',
]


def load_data_lab1(file_name: str) -> pd.DataFrame:
    data_file = DATA_DIR / file_name

    df = pd.read_csv(data_file)

    return df


def export(df: pd.DataFrame, name: str) -> None:
    EXPORT_DIR.mkdir(exist_ok=True, parents=True)

    name = name.split('.')[0]

    df.to_json(EXPORT_DIR / f'{name}.json')
    df.to_xml(EXPORT_DIR / f'{name}.xml')
    df.to_csv(EXPORT_DIR / f'{name}.csv')
    df.to_excel(EXPORT_DIR / f'{name}.xslx')


def describe_data(df: pd.DataFrame, name: str):
    print(f'File {name}')
    print(df.describe().loc[['25%', '50%', '75%']])


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    stds = df.std() * 3
    means = df.mean()
    lower_bound, upper_bound = means - stds, means + stds

    mask = (df > lower_bound) & (df < upper_bound)
    df_clean = df[mask.all(axis=1)]

    return df_clean


def r2_compare(df: pd.DataFrame) -> None:
    res = linregress(x=df['x'], y=df['y'])
    print(f'R2 before: {res.rvalue ** 2}')
    df_clean = drop_outliers(df)
    res_clean = linregress(x=df['x'], y=df['y'])
    print(f'R2 after: {res_clean.rvalue ** 2}')


def generate_data() -> np.ndarray:
    x = np.linspace(0, 5, 100)
    mu = 0
    sigma = 0.1
    noise = mu + sigma * np.random.randn(100)
    y = np.log(x + 1) + noise

    return y


# %%
for f_name in FILES:
    df = load_data_lab1(f_name)
    print(50 * '=')
    print(df)
    export(df, f_name)
    describe_data(df, f_name)
    r2_compare(df)
# %%
y = generate_data()
# %%
sns.scatterplot(x=np.linspace(0, 5, 100), y=y)

# %%
