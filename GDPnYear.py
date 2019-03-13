import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#load data set
url = "https://raw.githubusercontent.com/ParthanOlikkal/Regression/master/china_gdp.csv"
df = pd.read_csv(url)

#look at the data
df.head(10)

#plotting the dataset
plt.figure(figsize = (8,5))
x_data, y_data = (df["Year"].values, df["Values"].values)
plt.plot(x_data, y_data, 'ro')
plt.x_label('Year')
plt.y_label('GDP')
plt.show()

#after inspecting visually and identifying the plot it could be understood that it resembles a logistic function
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0/(1.0 + np.exp(-X))

plt.plot(X,Y)
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()


#Building the model
def sigmoid(x, Beta_1, Beta_2)					#Beta_1 : Controls the curve steepness
	y = 1 / (1 + np.exp(-Beta_1 * (x-Beta_2)))		#Beta_2 : Slides the curve on the x-axis
	return y

#giving a value to sigmoid function that might fit the data
beta_1 = 0.10
beta_2 = 1990.0

#Logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

#plot the initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

#Normalize the data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

#curve-fit uses non-linear least squares to fit the sigmoid function.
#optimal values of the parameters are minimized so that the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is less
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

#print the final parameters
print(" beta_1 = %f, beta_2 = %f " %(popt[0], popt[1]))

#plot the resulting regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize = (8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label = 'data')
plt.plot(x,y, linewidth = 3.0, label = 'fit')
plt.legend(loc = 'best')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

#Accuracy

#split the dataset into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

#build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, sigmoid_y)

#predict using test set
y_hat = sigmoid(test_x, *popt)

#evaluation
print("Mean absolute error : %.2f " %np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f " %np.mean((y_hat - test_y) ** 2))
from skLearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat, test_y))


