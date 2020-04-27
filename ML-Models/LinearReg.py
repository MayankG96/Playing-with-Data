
import quandl as qd
import math,datetime,pickle
import numpy as np

from sklearn import preprocessing,svm,model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

qd.ApiConfig.api_key = "8WgSAScLaZ-XKgb_q8zT"
data = qd.get('WIKI/GOOGL')

#Keeping just relevant columns
data = data[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
print(len(data))

#introducing 2 new required columns
data['HL_Pct'] = (data['Adj. High'] - data['Adj. Low'])/data['Adj. Low']*100.0
data['CO_pct'] = (data['Adj. Close'] - data['Adj. Open'])/data['Adj. Open']*100.0

#getting the final required dataset

data = data[['Adj. Close','Adj. Volume','HL_Pct','CO_pct']]

forecast_col = 'Adj. Close'

'''ML algorithms does not work on null values. So need to fill the null value. One of the best way to treat null values without'
loss of information is to fill it with an outlier(a really large +ve or -ve value)'''

data.fillna(-99999,inplace=True)

forecast_out = int(math.ceil(0.01*len(data)))

#creating our label , which will be close price of the stock sometime in future.This is what we aim to predict
data['future_CP'] = data[forecast_col].shift(-forecast_out)


#Separating features and label into different dataframes. Make sure length of both are same
x = np.array(data.drop(['future_CP'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:len(data)-forecast_out]
data.dropna(inplace=True)
y = np.array(data['future_CP'])


print(len(x),len(x_lately),len(y))
#Applying cross validation to split training and test data
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

#Training data will be used to fit our classifier

clf = LinearRegression()
clf.fit(x_train,y_train)




#Applying the classifier on test data, to check the accuracy'
acc = clf.score(x_test,y_test)

#forecasting future values/trend
forecast_set = clf.predict(x_lately)
print(forecast_set,acc,forecast_out)
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]




data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


