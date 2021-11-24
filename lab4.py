# %%
from pathlib import Path

import pandas as pd

# %%
SEPTAL_LENGTH = 'septal length'
SEPTAL_WIDTH = 'septal width'
PETAL_LENGTH = 'petal length'
PETAL_WIDTH = 'petal width'
DOMAIN = 'domain'
DATA_FILE = './data_final/iris.data'


def load_data() -> pd.DataFrame:
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        raise FileNotFoundError(f'File not found at {data_file.absolute()}')

    return pd.read_csv(
        data_file,
        names=[
            SEPTAL_LENGTH,
            SEPTAL_WIDTH,
            PETAL_LENGTH,
            PETAL_WIDTH,
            DOMAIN,
        ],
    )

# %%
data = load_data()
# %%
data.cov()
# %%
data.corr(method='pearson')
# %%
data.corr(method='spearman')
# %%
