import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

boston=pd.read_csv('train.csv')
boston.head()
print(boston)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['medv'], bins=100)
plt.show()
