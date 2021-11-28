# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from lab4 import load_data

RND_SEED = 42
FOLDS = 5
REPEATS = 1

MODELS = {
    'Regresja liniowa': LinearRegression(),
    'Regresja grzbietowa a=0.1': Ridge(random_state=RND_SEED, alpha=0.1),
    'Regresja grzbietowa a=0.5': Ridge(random_state=RND_SEED, alpha=0.5),
    'Regresja grzbietowa a=1': Ridge(random_state=RND_SEED, alpha=1),
    'Regresja grzbietowa a=2': Ridge(random_state=RND_SEED, alpha=2),
    'Regresja LASSO a=0.1': Lasso(random_state=RND_SEED, alpha=0.1),
    'Regresja LASSO a=0.5': Lasso(random_state=RND_SEED, alpha=0.5),
    'Regresja LASSO a=1': Lasso(random_state=RND_SEED, alpha=1),
    'Regresja LASSO a=2': Lasso(random_state=RND_SEED, alpha=2),
    'Perceptron wielowarstwowy': MLPRegressor(random_state=RND_SEED),
    'SVM': SVR(),
}

sns.set()


def add_noise(data: np.ndarray) -> np.ndarray:
    noise = np.random.rand(*data.shape) - 0.5

    return data + noise


def preprocess(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()

    return X, y


def run_training(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list]:
    rkf = RepeatedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=RND_SEED)
    scores = np.zeros((len(MODELS), FOLDS * REPEATS))
    models = []

    for fold_id, (train_idx, test_idx) in enumerate(rkf.split(X, y)):
        for model_id, model_name in enumerate(MODELS):
            model = clone(MODELS[model_name])
            model.fit(X[train_idx], y[train_idx])
            y_hat = model.predict(X[test_idx])

            scores[model_id, fold_id] = r2_score(y[test_idx], y_hat)
            if fold_id == 0:
                models.append(model)

    return scores, models


data = load_data()
X, y = preprocess(data)
# Default training
res, models = run_training(X, y)
print('Wyniki')
print(res.mean(axis=1))
# Noise
X_noise = add_noise(X)
res_noise, models_noise = run_training(X_noise, y)
print('Wyniki po dodaniu szumu')
print(res_noise.mean(axis=1))
print('Korelacja Pearsona')
print(data.corr(method='pearson'))
print('Korelacja Spearmana')
print(data.corr(method='spearman'))
print('Kowariancja')
print(data.cov())
lr = LinearRegression()
lr.fit(X, y)
print('Regresja liniowa dla całego zbioru')
print(f'R2: {lr.score(X, y)}')
print(f'Współczynniki: {lr.coef_}')
sns.pairplot(data)
plt.show()
plt.clf()
# Ridge
print('Grzbietowa współczynniki')
print(models[1].coef_)
print(models[2].coef_)
print(models[3].coef_)
print(models[4].coef_)
# Lasso
print('Lasso współczynniki')
print(models[5].coef_)
print(models[6].coef_)
print(models[7].coef_)
print(models[8].coef_)
print('R2 dla modeli')
for m, r in zip(MODELS.keys(), res.mean(axis=1)):
    print(f'{m:30}: {r}')
# All
sns.boxplot(data=res.T)
plt.show()
plt.clf()
# Ridge
sns.boxplot(data=res[1:5].T)
plt.show()
plt.clf()
# Lasso
sns.boxplot(data=res[5:9].T)
plt.show()
plt.clf()
# Lasso compare
sns.boxplot(data=np.concatenate((res[5:9].T, res_noise[5:9].T), axis=1))
plt.show()
plt.clf()
# %%
# Wczytanie
res = np.load('lab4_res', allow_pickle=True)
res_noise = np.load('lab4_res_noise', allow_pickle=True)
