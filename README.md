# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize 𝑘 k cluster centroids randomly from the dataset.
2. Assign each customer (data point) to the nearest centroid based on the Euclidean distance.
3. Update each centroid to be the mean of all points assigned to it.
4. Repeat steps 2 and 3 until centroids no longer change significantly or reach a set number of iterations.
5. Label each customer with their respective cluster for segmentation analysis.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Chithradheep R
RegisterNumber:  2305002003

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)','Spending Score (1-100)']]
plt.figure(figsize=(4,4))
plt.scatter(x["Annual Income (k$)"],x["Spending Score (1-100)"])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=3
kmeans=KMeans(n_clusters=k)
kmeans.fit(x)
centroids=kmeans.cluster_centers_
labels = kmeans.labels_
print("centroids:")
print(centroids)
print("labels:")
print(labels)
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_data = x[labels == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], c=colors[i])
    distance = euclidean_distances(cluster_data, [centroids[i]])
    radius = np.max(distance)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='k', label='Centroids')
plt.title('k-means clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
*/
```

## Output:
![Screenshot 2024-11-07 095851](https://github.com/user-attachments/assets/ace57f98-3424-4c8f-ae4f-b06824f71bef)
![Screenshot 2024-11-07 095911](https://github.com/user-attachments/assets/eb962a5a-47f4-488c-862b-01cbd12cd92a)
![Screenshot 2024-11-07 095932](https://github.com/user-attachments/assets/c77e2323-d304-4203-9d1f-4a7d27b321db)
![Screenshot 2024-11-07 095951](https://github.com/user-attachments/assets/0ab85a29-017d-4486-bac8-049eb6dd24bf)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
