
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
file1 = "long.pkl"
file2 = "lat.pkl"

# Importing the dataset
dataset = pd.read_csv('suyogDataset.csv')

X = dataset.iloc[:, 3:].values
ylat = dataset.iloc[:, 0].values
ylon = dataset.iloc[:, 1].values
# X[X==0] = -200



from sklearn.model_selection import train_test_split
X_train, X_test, ylat_train, ylat_test = train_test_split(X, ylat, test_size = 0.01, random_state = 33)

ylon_train, ylon_test = train_test_split(ylon,test_size = 0.01,random_state = 33)

print("xte",X_test)



# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor1 = LinearRegression()

# regressor.fit(X_train, ylat_train)
# regressor1.fit(X_train, ylon_train)

# pickle.dump(regressor,open(file2,'wb'))
# pickle.dump(regressor1,open(file1,'wb'))

# Predicting the Test set results

lat_mod = pickle.load(open(file2,'rb'))
lon_mod = pickle.load(open(file1,'rb'))

ylat_pred = lat_mod.predict(X_test[0])
ylon_pred = lon_mod.predict(X_test[0])
print("lat test",ylat_test)
print("lat pred",type(ylat_pred))
print("lon test",ylon_test)
print("lon pred",ylon_pred)

