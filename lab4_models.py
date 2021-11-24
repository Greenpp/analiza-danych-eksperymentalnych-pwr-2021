# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from lab4 import (
    DOMAIN,
    PETAL_LENGTH,
    PETAL_WIDTH,
    SEPTAL_LENGTH,
    SEPTAL_WIDTH,
    load_data,
)

RND_SEED = 42
FOLDS = 5
REPEATS = 1

MODELS = {
    'Regresja liniowa': LinearRegression(),
    'Regresja grzbietowa': Ridge(random_state=RND_SEED),
    'Regresja LASSO': Lasso(random_state=RND_SEED),
    'Perceptron wielowarstwowy': MLPRegressor(random_state=RND_SEED),
    'SVM': SVR(),
}

sns.set()


def add_noise(data: np.ndarray) -> np.ndarray:
    noise = np.random.rand(*data.shape) - 0.5

    return data + noise


def preprocess(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].factorize()[0]

    return X, y


def run_training(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rskf = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=REPEATS, random_state=RND_SEED
    )
    scores = np.zeros((len(MODELS), FOLDS * REPEATS))

    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        for model_id, model_name in enumerate(MODELS):
            model = clone(MODELS[model_name])
            model.fit(X[train_idx], y[train_idx])
            y_hat = model.predict(X[test_idx])

            scores[model_id, fold_id] = accuracy_score(y[test_idx], y_hat)

    return scores


# %%
data = load_data()
# %%
X, y = preprocess(data)
# %%
res = run_training(X, y)
# %%
from sklearn.datasets import load_diabetes
# %%
data = load_diabetes()
# %%
data['data'].shape
# %%
