from tkinter import X

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import pyod
from pyod.models.cblof import CBLOF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("patient_records_500.xls")
xyyy=print(df['Temperature'].describe())
inputdata = df.drop(['Haccl_x','Haccl_y','Haccl_z','Baccl_x','Baccl_y','Baccl_z','Time','X_change','Y_change','Z_change','Outcome','counter','Average_accl','Case1','Case2','Case3','Case4','Case5'],axis=1)
print(inputdata.describe())
print(inputdata.info())


#Describes the dataset
#Provides important Information such as Skewness and Kurtosis on the features
print("Skewness of Temperature: %f" % df['Temperature'].skew())
print("Kurtosis of Temperature: %f" % df['Temperature'].kurt())
print("Skewness of Accelerometer: %f" % df['Total_change'].skew())
print("Kurtosis of Accelerometer: %f" % df['Total_change'].kurt())
change = print(df['Total_change'].describe())




#This will plot the scatter of different cases
ax = plt.gca()
df.plot(kind='line',x='counter',y='Case1',ax=ax,)
df.plot(kind='line',x='counter',y='Case2', color='red', ax=ax, )
df.plot(kind='line',x='counter',y='Case3', color='green', ax=ax, )
df.plot(kind='line',x='counter',y='Case4', color='orange', ax=ax, )
df.plot(kind='line',x='counter',y='Case5', color='purple', ax=ax, )

plt.show()


