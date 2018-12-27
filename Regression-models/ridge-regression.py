import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# ---- Load the data set ----
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
# print(boston_df.info())

# --- Change target to Price ---
boston_df['Price']=boston.target

# print boston_df.head()

X = boston_df.drop('Price',axis = 1)
# print newX[0:3] # check
y = boston_df['Price']

# ---- Split 75:25 -----
X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size = 0.25, random_state = 3)

# ---- Fit the model -----
lr = LinearRegression()
lr.fit(X_train, y_train)

# ----- Apply Ridge Regression ------
# Higher the alpha value, more restriction on the coefficients
# One with low alpha value
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)
# Another with very high alpha
rr100 = Ridge(alpha=100)
rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)
test_score=lr.score(X_test, y_test)

Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score100 = rr100.score(X_train,y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

print("linear regression train score:", train_score)
print("linear regression test score:", test_score)
print("ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)
print("ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100)

plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers

plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency

plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show(block = False)
# plt.savefig('Ridge-Regression.png')