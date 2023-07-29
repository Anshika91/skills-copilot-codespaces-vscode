#!/usr/bin/env python
# coding: utf-8

# # Clustering - K-Means and Hierachial
# 
# In the last notebook, we learned that data can be broken into clusters and learned how to see whether data may be compatible with such an analysis. In this notebook, we will perform this clustering automatically.
# 
# To get started, run the cell below to load our data
# 
# > **Citation**: The seeds dataset used in the this exercise was originally published by the Institute of Agrophysics of the Polish Academy of Sciences in Lublin, and can be downloaded from the UCI dataset repository (Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science).

# In[ ]:


import pandas as pd

# load the training dataset
get_ipython().system('wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/seeds.csv')
data = pd.read_csv('seeds.csv')

# Display a random sample of 10 observations (just the features)
features = data[data.columns[0:6]]
features.sample(10)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Normalize the numeric features so they're on the same scale
scaled_features = MinMaxScaler().fit_transform(features[data.columns[0:6]])

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
features_2d[0:10]


# ## K-Means Clustering
# 
# The algorithm we used to create our test clusters is *K-Means*. This is a commonly used clustering algorithm that separates a dataset into *K* clusters of equal variance. The number of clusters, *K*, is user defined. The basic algorithm has the following steps:
# 
# 1. A set of K centroids are randomly chosen.
# 2. Clusters are formed by assigning the data points to their closest centroid.
# 3. The means of each cluster is computed and the centroid is moved to the mean.
# 4. Steps 2 and 3 are repeated until a stopping criteria is met. Typically, the algorithm terminates when each new iteration results in negligable movement of centroids and the clusters become static.
# 5. When the clusters stop changing, the algorithm has *converged*, defining the locations of the clusters - note that the random starting point for the centroids means that re-running the algorithm could result in slightly different clusters, so training usually involves multiple iterations, reinitializing the centroids each time, and the model with the best WCSS is selected.
# 
# Let's try using K-Means on our seeds data with a K value of 3.

# In[ ]:


from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters


# Let's see those cluster assignments with the two-dimensional data points.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)


# Hopefully, the data has been separated into three distinct clusters.
# 
# So what's the practical use of clustering? In some cases, you may have data that you need to group into distinct clusters without knowing how many clusters there are or what they indicate. For example, a marketing organization might want to separate customers into distinct segments, and then investigate how those segments exhibit different purchasing behaviors.
# 
# Sometimes, clustering is used as an initial step towards creating a classification model. You start by identifying distinct groups of data points, and then assign class labels to those clusters. You can then use this labelled data to train a classification model.
# 
# In the case of the seeds data, the different species of seed are already known and encoded as 0 (*Kama*), 1 (*Rosa*), or 2 (*Canadian*), so we can use these identifiers to compare the species classifications to the clusters identified by our unsupervised algorithm.

# In[ ]:


seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)


# There may be some differences between the cluster assignments and class labels, but the K-Means model should have done a reasonable job of clustering the observations so that seeds of the same species are generally in the same cluster.

# ## Hierarchical Clustering
# 
# Hierarchical clustering methods make fewer distributional assumptions when compared to K-means methods. However, K-means methods are generally more scalable, sometimes very much so.
# 
# Hierarchical clustering creates clusters by either a *divisive* method or *agglomerative* method. The divisive method is a "top down" approach starting with the entire dataset and then finding partitions in a stepwise manner. Agglomerative clustering is a "bottom up" approach. In this lab you will work with agglomerative clustering which roughly works as follows:
# 
# 1. The linkage distances between each of the data points is computed.
# 2. Points are clustered pairwise with their nearest neighbor.
# 3. Linkage distances between the clusters are computed.
# 4. Clusters are combined pairwise into larger clusters.
# 5. Steps 3 and 4 are repeated until all data points are in a single cluster.
# 
# The linkage function can be computed in a number of ways:
# - Ward linkage measures the increase in variance for the clusters being linked,
# - Average linkage uses the mean pairwise distance between the members of the two clusters,
# - Complete or Maximal linkage uses the maximum distance between the members of the two clusters.
# 
# Several different distance metrics are used to compute linkage functions:
# - Euclidian or l2 distance is the most widely used. This metric is only choice for the Ward linkage method.
# - Manhattan or l1 distance is robust to outliers and has other interesting properties.
# - Cosine similarity, is the dot product between the location vectors divided by the magnitudes of the vectors. Notice that this metric is a measure of similarity, whereas the other two metrics are measures of difference. Similarity can be quite useful when working with data such as images or text documents.
# 
# ### Agglomerative Clustering
# 
# Let's see an example of clustering the seeds data using an agglomerative clustering algorithm.

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters


# So what do the agglomerative cluster assignments look like?

# In[ ]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, agg_clusters)


# ## Summary
# 
# Here we practiced using K-means and hierarchical clustering. This unsupervised learning has the ability to take unlabelled data and identify which of these data are similar to one another. 
# 
# ## Further Reading
# 
# To learn more about clustering with scikit-learn, see the [scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html).
# 
