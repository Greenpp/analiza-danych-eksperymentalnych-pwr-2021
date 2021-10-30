# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

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

    print('Exporting to json')
    df.to_json(EXPORT_DIR / f'{name}.json')
    print('Exporting to xml')
    df.to_xml(EXPORT_DIR / f'{name}.xml')
    print('Exporting to csv')
    df.to_csv(EXPORT_DIR / f'{name}.csv')
    print('Exporting to xslx')
    df.to_excel(EXPORT_DIR / f'{name}.xslx')


def describe_data(df: pd.DataFrame, name: str):
    desc = df.describe()
    print(f'1 kwartyl:\n{desc.loc["25%"]}')
    print(f'mediana:\n{desc.loc["50%"]}')
    print(f'3 kwartyl:\n{desc.loc["75%"]}')


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    stds = df.std() * 3
    means = df.mean()
    lower_bound, upper_bound = means - stds, means + stds

    mask = (df > lower_bound) & (df < upper_bound)
    df_clean = df[mask.all(axis=1)]

    return df_clean


def r2_compare(df: pd.DataFrame) -> None:
    res = linregress(x=df['x'], y=df['y'])
    print(f'R2 przed odrzuceniem odstających wartości: {res.rvalue ** 2}')
    df_clean = drop_outliers(df)
    res_clean = linregress(x=df_clean['x'], y=df_clean['y'])
    print(f'R2 po odrzuceniu odstających wartości: {res_clean.rvalue ** 2}')


def generate_1d_data(plot: bool = False) -> np.ndarray:
    x = np.linspace(0, 5, 100)
    mu = 0
    sigma = 0.1
    noise = mu + sigma * np.random.randn(100)
    y = np.log(x + 1) + noise

    if plot:
        sns.scatterplot(x=x, y=y)
        plt.show()
        plt.clf()

    return y


def generate_2d_data(plot: bool = False) -> np.ndarray:
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)

    z = np.sin(np.log(x + 2) * np.sqrt(y))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f')

        ax.scatter(x, y, z)
        plt.show()
        plt.clf()

    return z


def calculate_coeff(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    p = Pipeline(
        [
            ('poly', PolynomialFeatures(degree)),
            ('lr', LinearRegression(fit_intercept=False)),
        ]
    )

    p.fit(x, y)
    coeff = p.named_steps["lr"].coef_
    print(f'Współczynniki: {coeff}')

    return coeff


def get_design_matrix(data: np.ndarray, degree: int) -> None:
    if len(data.shape) == 1:
        data = data.reshape((-1, 1))

    poly = PolynomialFeatures(degree)
    m = poly.fit_transform(data)

    print(m)


def plot_poly(data: np.ndarray, coeff: np.ndarray) -> None:
    l = np.linspace(0, 5, 100)
    y = sum([l ** i * n for i, n in enumerate(coeff)])

    sns.scatterplot(x=l, y=data)
    sns.lineplot(x=l, y=y, color='orange')
    plt.show()
    plt.clf()


# %%
for f_name in FILES:
    print(50 * '=')
    print(f'FILE: {f_name}')
    df = load_data_lab1(f_name)
    print(df)
    export(df, f_name)
    describe_data(df, f_name)
    r2_compare(df)


print('FUNKCJA 1 ZMIENNEJ')
data_1d = generate_1d_data(plot=True)
l = np.linspace(0, 5, 100).reshape((-1, 1))

print('WIELOMIAN 1 STOPNIA')
print('[1, a]')
coeff_1 = calculate_coeff(l, data_1d, 1)
plot_poly(data_1d, coeff_1)
print('WIELOMIAN 3 STOPNIA')
print('[1, a, a^2, a^3]')
coeff_3 = calculate_coeff(l, data_1d, 3)
plot_poly(data_1d, coeff_3)


print('FUNKCJA 2 ZMIENNYCH')
data_2d = generate_2d_data(plot=True)
print('[1, a, b, a^2, ab, b^2]')
coeff_2 = calculate_coeff(np.hstack((l, l)), data_2d, 2)

l = l.reshape(-1)
reg_2d = (
    coeff_2[0]
    + coeff_2[1] * l
    + coeff_2[2] * l
    + coeff_2[3] * l ** 2
    + coeff_2[4] * l * l
    + coeff_2[5] * l ** 2
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')

ax.scatter(l, l, data_2d)
ax.plot(l, l, reg_2d, color='orange')
plt.show()
plt.clf()

# %%
