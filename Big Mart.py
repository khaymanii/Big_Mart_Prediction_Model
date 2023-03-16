
# Importing the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Data collection and analysis

big_mart_data = pd.read_csv('Train.csv')
big_mart_data.head()
big_mart_data.info()
big_mart_data.isnull().sum()
big_mart_data.shape


# Handling missing values

# Mean value of item weight column

big_mart_data['Item_Weight'].mean()


# Filling the missing values in "Item weight" column with mean value

big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
big_mart_data.isnull().sum()


# Replacing the missing values in "Outlest size" with mode

mode = big_mart_data.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x: x.mode()[0]))

print(mode)


missing_values = big_mart_data['Outlet_Size'].isnull()
print(missing_values)
big_mart_data.loc[missing_values, 'Outlet_Size'] = big_mart_data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode)
big_mart_data.isnull().sum()


# Data Analysis

big_mart_data.describe()


# Numerical Features Plot

sns.set()


# Item_Weight distribution

plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()


# Distribution of Item Visibility

plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()


# Item MRP Distribution

plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()


# Item outlet sales distribution

plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()


# Outlet Establishment Year Countplot

plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()


# Distribution of Categorical Features

plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()


# Item Type Column Distribution

plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()


# Data Preprocessing

big_mart_data.head()
big_mart_data['Item_Fat_Content'].value_counts()

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
big_mart_data['Item_Fat_Content'].value_counts()


# label Encoding

encoder = LabelEncoder()


big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'], )

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])



# Splitting Features and Target

X = big_mart_data.drop(columns = 'Item_Outlet_Sales', axis=1)
y = big_mart_data['Item_Outlet_Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


model = LogisticRegression()

model.fit(X_train, y_train)


# Evaluation on taining data

training_data_prediction = model.predict(X_train)

r2_train = metrics.r2_score(y_train, training_data_prediction)

print('R Squared value :', r2_train)


# Evaluation on test data

test_data_prediction = model.predict(X_test)

r2_test = metrics.r2_score(y_test, test_data_prediction)

print('R Squared value :', r2_test)