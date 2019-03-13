import matplotlib.pyplot as plt
import pandas as pd
import pyplot as pl
import numpy as np
%matplotlib inline

url = "https://raw.githubusercontent.com/ParthanOlikkal/CO2-emission-of-cars/master/FuelConsumptionCo2.csv"

#Load the data
df = pd.read_csv("FuelConsumption.csv")

#Look at the data
df.head()

#summarize the data
df.describe()

#selecting different features
cdf = df[['ENGINESIZE','CYLINDER','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#plotting the data
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color = "blue")
plt.xlabel("Fuelconsumption Comb")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = "red")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDER, cdf.CO2EMISSIONS, color = "green")
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()

#splitting testing and training models
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSION, color = "blue")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Model data
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(tain[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

#The coefficients
print (Coefficients: ',regr.coef_)
print ('Intercept: ', regr.intercept_)

#plot the fit line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = "blue")
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluation
from sklearn.metrics import r2_score

text_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" %np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
