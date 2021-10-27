# %%
from pathlib import Path
import seaborn as sns

import pandas as pd

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
    print(df.describe([0.25, 0.5, 0.75]))


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    stds = df.std() * 3
    means = df.mean()
    lower_bound, upper_bound = means - stds, means + stds

    mask = (df > lower_bound) & (df < upper_bound)
    df_clean = df[mask.all(axis=1)]
    
    return df_clean


# %%
for f_name in FILES:
    df = load_data_lab1(f_name)
    export(df, f_name)
    describe_data(df, f_name)
# %%
