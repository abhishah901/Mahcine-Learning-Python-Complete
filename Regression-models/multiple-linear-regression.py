
# ----library imports-----
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ---- import a simple dataset with 2 numeric columns ----------
data = pd.read_csv('data/Economy.csv')

# --- look at the data ----
print(data.shape)
print(data.head())
print(data.describe())

# ----- Prepare data for split, train and fitting purposes -----
temp = (len(data.columns) - 1)
# print(temp)
# print(type(temp))
X = data.iloc[:,:temp]
y = data.iloc[:,-1]

# ----- Split > 80:20 -----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ------ Fit ---------
Mreg = LinearRegression()
Mreg.fit(X_train, y_train)

# ----- Coefficients ------
coeffs = pd.DataFrame(Mreg.coef_, X.columns, columns=['Coefficient'])
print(coeffs)

# ----- Predict --------
y_pred = Mreg.predict(X_test)

# ----- Checking Actual vs Predicted -----
holder = pd.DataFrame({'Real': y_test, 'Predictions': y_pred})
print(holder)

# ------ Evaluation and Metrics -----
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
