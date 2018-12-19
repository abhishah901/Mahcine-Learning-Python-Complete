# imports

import pandas as pd  # to read the data into a dataframe
import matplotlib.pyplot as plt # basic plotting
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, scale


# Data Source: https://www.analyticsvidhya.com/wp-content/uploads/2016/07/loan_prediction-1.zip


# Import train data

X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')

# Import test data. Not entirely important, but let's do it for the sake of it.

X_test = pd.read_csv('data/X_test.csv')
Y_test = pd.read_csv('data/Y_test.csv')

# Look at the training data
print("Head without any pre-processing:")
print(X_train.head())
print("Description of numeric variables:")
print(X_train.describe())

# collect numeric data and plot histograms for the same
# X_train[X_train.dtypes[(X_train.dtypes == 'float64') | (X_train.dtypes == 'int64')].index.values].hist()
# plt.show()  # Plot the histogram values


# ---------Min-max Scaling--------
X_train_minmax = MinMaxScaler().fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# for the sake of it...
X_test_minmax = MinMaxScaler().fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

# Look at the scaled data
print("Scaled Head:")
print(X_train.head())
print("Mean: \n".format(X_train_minmax.scale_(axis = 0)))


# -----------Label Encoding-------
hold_func = LabelEncoder()
for col in X_test.columns.values:
    # Encoding only categorical(non-numeric) variables
    if X_test[col].dtypes == 'object':
        data = X_train[col].append(X_test[col])  # temporary holder for data
        hold_func.fit(data.values)  # fit encoder
        X_train[col] = hold_func.transform(X_train[col])  # replacing the columns with with encoded ones in train
        X_test[col] = hold_func.transform(X_test[col])    # as well as test


# Look at the label-encoded data
print("Label Encoded Head:")
print(X_train.head())

# --------- One Hot Encoding-------
hold_func = OneHotEncoder(sparse=False)

X_train_1 = X_train

X_test_1 = X_test


columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

for col in columns:
    # collect categorical columns
    data = X_train[[col]].append(X_test[[col]])
    hold_func.fit(data)
    # Fitting One Hot Encoding on train data
    temp = hold_func.transform(X_train[[col]])
    # Renaming new columns
    temp = pd.DataFrame(temp, columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
    # In side by side concatenation index values should be same
    # Setting the index values similar to the X_train data frame
    temp = temp.set_index(X_train.index.values)
    # adding the new One Hot Encoded varibales to the train data frame
    X_train_1 = pd.concat([X_train_1, temp], axis=1)
    # fitting One Hot Encoding on test data
    temp = hold_func.transform(X_test[[col]])
    # changing it into data frame and adding column names
    temp = pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
    # Setting the index for proper concatenation
    temp = temp.set_index(X_test.index.values)
    # adding the new One Hot Encoded variables to test data frame
    X_test_1 = pd.concat([X_test_1, temp], axis=1)


# Look at the One-Hot Encoded data

print("One-Hot Encoded Head:")
print(X_train_1.head())
