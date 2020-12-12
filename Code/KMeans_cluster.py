# Modules
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# Import the data
df = pd.read_csv('file_for_KClusters.csv')
#printing entire input file
print(df.head())
#printing column 1 which is temperature
print(df.iloc[:,0])
#assigning respective cells to a new variable
x=df.iloc[:,[0,1,2,3,4,5,6]].values
print(x)
kmeans5= KMeans(n_clusters=3)
y_kmeans5 = kmeans5.fit_predict(x)

#Plotting Elbow Curve to find out the optimal number of clusters

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error, marker='o')
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
print(y_kmeans5)
centroids = kmeans5.cluster_centers_
print(centroids)
fig, ax = plt.subplots(figsize=(6, 6))

#plt.scatter(x[y_kmeans5==4,0],x[y_kmeans5==4,1],s=30, c='magenta', edgecolor='black', label = 'cluster1', marker='s')
plt.scatter(x[y_kmeans5==1,0],x[y_kmeans5==1,1],s=30, c='orange', edgecolor='black', label = 'cluster1', marker='o')
plt.scatter(x[y_kmeans5==2,0],x[y_kmeans5==2,1],s=30, c='green', edgecolor='black', label='cluster2', marker='+')
plt.scatter(x[y_kmeans5==0,0],x[y_kmeans5==0,1],s=30, c='blue', edgecolor='black', label = 'cluster3', marker='8')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='r', label='centroid', edgecolors='black')

plt.legend()
plt.xlim([35.5, 39.5])
plt.ylim([-100, 400])
plt.xlabel('Temperature')
plt.ylabel('Change in Accelerometer')
plt.title('K-Means clusters for Patient Data', fontweight='bold')
#ax.set_aspect('equal')
plt.show()