import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.datasets import make_blobs
from numpy import random, where

#currently this is running on Credit Card dataset, maybe I can transform
#this and use it on my own dataset
from dbscan2_credit_card import cvec
gc.collect()
X = pd.read_csv('patient_records_dbscan.csv')


#dropping unnecessary columns
X = X.drop(['Haccl_x','Haccl_y','Haccl_z','Baccl_x','Baccl_y','Baccl_z','Time','X_change','Y_change','Z_change','Total_change','Outcome','counter','Average_accl'],axis=1)
#Converting Gender into binary values
X.Gender[X.Gender == 'Male'] = 1
X.Gender[X.Gender == 'Female'] = 0
print(X.head())
print(X.info()) #finding how large the dataset is
print(X.describe()) #finding min, max, mean and STD
print(X[X.isna().any(axis=1)])

# Scaling the data to bring all the attributes to a comparable level
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("The Scaled value for this dataset looks like this")
print(X_scaled)

# Normalizing the data so that
# the data approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)
print("The normalized dataset looks like this")
print(X_normalized)


# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)
# Performing PCA analysis to reduce dataset
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())
# Show results
#print('Original number of features:', X.shape[1])
#print('Reduced number of features:', X_principal.shape[1])

dbscan = DBSCAN(eps=0.4, metric='euclidean',min_samples=10)
dbscan.fit(X_normalized)
print("This is the output from the DBSCAN")
print(dbscan.labels_)

labels=dbscan.labels_
print("Now trying to print Silhouette scores")

#Calculating core points.
core_samples = np.zeros_like(labels, dtype=bool)
core_samples[dbscan.core_sample_indices_] = True
print(core_samples)

pca=PCA(n_components=2).fit(X_normalized)
pca_3d = pca.transform(X_normalized)
#print("This is the output from the PCA")
#print(pca_3d)
fig5 = plt.gcf()

#Plotting different cluster scores for the provided dataset
for i in range(0,pca_3d.shape[0]):
    #The values from the scatter matrix can be changed to plot samples for all clusters
    #with different epsilon values and sample sizes
    #e.g. pca_3d[i,1],pca_3d[i,1]
    # e.g. pca_3d[i,2],pca_3d[i,1]
    if dbscan.labels_[i]==0:
        p1 = plt.scatter(pca_3d[i,0],pca_3d[i,1],c='k',marker='+')
    elif dbscan.labels_[i]==1:
        p2 = plt.scatter(pca_3d[i,0],pca_3d[i,1],c='g',marker='o')
    elif dbscan.labels_[i]==-1:
        p3 = plt.scatter(pca_3d[i,0],pca_3d[i,1],c='b',marker='v')

#This legend is for clusters
plt.legend([p1, p2, p3], ['Temperature', 'Accelerometer','BMI'])

plt.title('DBSCAN clusters: Epsilon = 4 , Min. Samples = 10 ')
plt.plot()
plt.show()
plt.draw()
fig5.savefig('eps4_sam10.png', dpi=300)