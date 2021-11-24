# %%
from pathlib import Path

import pandas as pd

# %%
STRENGTH = 'strength'
TEMPERATURE = 'temperature'
PRESSURE = 'pressure'
DATA_FILE = './data_final/plastic.dat'


def load_data() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f'File not found at {data_file.absolute()}')

    return pd.read_csv(data_file, names=[STRENGTH, TEMPERATURE, PRESSURE], skiprows=7)


# %%
data = load_data()
# %%
data.cov()
# %%
data.corr(method='pearson')
# %%
data.corr(method='spearman')
# %%
data
# %%
