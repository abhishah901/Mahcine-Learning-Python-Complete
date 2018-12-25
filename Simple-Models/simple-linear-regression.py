
# ----library imports-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ---- import a simple dataset with 2 numeric columns ----------
data = pd.read_csv('data/student_scores.csv')

# --- look at the data ----
print('Shape:', data.shape)
print('Head:', data.head())
print('Description:', data.describe())

# ---- Simple plot ------
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show(block=False)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, 1].values,
                                                    test_size=0.2,
                                                    random_state=0)

Lreg = LinearRegression()
Lreg.fit(X_train, y_train)

# ---- Some interesting Model Attributes -----
print('Regression Intercept:', Lreg.intercept_)
print('Regression Coefficient:', Lreg.coef_)

# ------ Apply model -------
y_pred = Lreg.predict(X_test)

# ----- Checking Actual vs Predicted -----
holder = pd.DataFrame({'Real': y_test, 'Predictions': y_pred})

# ------ Evaluation and Metrics -----
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
