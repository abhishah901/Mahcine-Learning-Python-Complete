import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---- Get data ------
datasets = pd.read_csv('data/Positions.csv')
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:, 2].values


# ---- Fitting Polynomial Regression -----

poly_reg = PolynomialFeatures(degree = 4)
X_Poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_Poly, Y)

# ---- Basic plot of the fit -----
X_Grid = np.arange(min(X), max(X), 0.1)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X,Y, color = 'red')
plt.plot(X_Grid, lin_reg_2.predict(poly_reg.fit_transform(X_Grid)), color = 'blue')
plt.title('Polynomial Regression results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show(block=False)

# plt.savefig('PolynomialFit.png')
