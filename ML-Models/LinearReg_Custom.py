#import all necessary libraries/modules
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#functions to create a dataset,calculating parameters(m,b) for best fit line and calculating R^2 value


def create_dataset(count,variance,step = 2,correlation = 'False'):
    val = 1
    ys = []
    if correlation and correlation == 'pos':
        for i in range(count):
            y = val + random.randrange(-variance,variance)
            ys.append(y)
            val += step

    elif correlation and correlation == 'neg':
        for i in range(count):
            y = val - random.randrange(-variance,variance)
            ys.append(y)
            val -= step

    else:
        for i in range(count):
            y = val + random.randrange(-variance, variance)
            ys.append(y)

    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)


def best_fit_slope(xs,ys):

    m = ((mean(xs) * mean(ys)) - (mean(xs * ys)))/\
        ((mean(xs)*mean(xs))-mean(xs*xs))
    b = mean(ys) - (m * mean(xs))
    return m, b

def r_squared(ys,ys_line):
    y_mean = mean(ys)
    SE_ytotal = sum((ys - y_mean) ** 2)
    SE_yline = sum((ys-ys_line)**2)

    r_square = 1 - (SE_yline/SE_ytotal)
    return r_square

xs,ys = create_dataset(40,10,4,)

m,b = best_fit_slope(xs,ys)

'calculate the regression line'
regression_line = []

for i in range(len(xs)):
    y = m*xs[i] + b
    regression_line.append(y)

'measure R^2 value'
fit_estimate = r_squared(ys,regression_line)
print(fit_estimate)

plt.plot(regression_line)
plt.scatter(xs,ys)
plt.show()

