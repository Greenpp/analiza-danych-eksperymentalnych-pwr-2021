# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
data.drop(columns="domain", inplace=True)
# %%
data
# %%
data.mean()
# %%
data.median()
# %%
data.var()
# %%
sns.histplot(data["septal length"])
# %%
sns.histplot(data["septal width"])
# %%
sns.histplot(data["petal length"])
# %%
sns.histplot(data["petal width"])
# %%
