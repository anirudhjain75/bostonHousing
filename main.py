import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

boston = pd.read_csv('train.csv')
boston = boston[boston.medv < 34]
boston = boston[boston.medv > 6]
print(boston.shape)
boston = boston.dropna()

y_train = boston['medv'].values.reshape(-1, 1)

reg=linear_model.LinearRegression()

boston = boston.set_index('ID')

print(boston.columns)

for value in boston.columns[:-1]:
    X_train = boston[value].values.reshape(-1, 1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    plt.scatter(x=y_train, y=y_pred)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('medv')
    plt.ylabel('Predicted from ' + value)
    plt.savefig("comp/" + value)
    plt.figure()

plt.show()
