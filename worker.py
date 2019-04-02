import numpy as np
import pandas as pd
import pickle
file1 = "long.pkl"
file2 = "lat.pkl"


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# dataset = pd.read_csv('suyogDataset.csv')

# X = dataset.iloc[:, 3:].values
# ylat = dataset.iloc[:, 0].values
# ylon = dataset.iloc[:, 1].values
# X[X==0] = -200

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# from sklearn.model_selection import train_test_split
# X_train, X_test, ylat_train, ylat_test = train_test_split(X, ylat, test_size = 0.01, random_state = 33)

# ylon_train, ylon_test = train_test_split(ylon,test_size = 0.01,random_state = 33)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor1 = LinearRegression()

# regressor.fit(X_train, ylat_train)
# regressor1.fit(X_train, ylon_train)

# pickle.dump(regressor,open(file2,'wb'))
# pickle.dump(regressor1,open(file1,'wb'))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


lat_mod = pickle.load(open(file2,'rb'))
lon_mod = pickle.load(open(file1,'rb'))

# +++++++++++++++++++++++++++++++++++++++++
# ylat_pred = lat_mod.predict(X_test)
# ylon_pred = lon_mod.predict(X_test)
# print("lat test",ylat_test)
# print("lat pred",ylat_pred)
# print("lon test",ylon_test)
# print("lon pred",ylon_pred)
# ++++++++++++++++++++++++++++++++++++++++++++

def calculate(str):
	rsssi = map(int,str.split(" "))
	rsssi = np.array(rsssi)
	ylat_pred = lat_mod.predict(rsssi)
	ylon_pred = lon_mod.predict(rsssi)
	position = {
		'lat':ylat_pred[0],
		'lon':ylon_pred[0]
	}
	return position