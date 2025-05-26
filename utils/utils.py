import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .KMeans import kmeans
import time

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    from sklearn.cluster import KMeans
    # sckit-learn xxxx (cuda problem)
    kmeans_sklearn = KMeans(n_clusters=n_mem, init='k-means++',n_init=10)
    temp = x.detach().cpu().numpy()
    kmeans_sklearn.fit(temp)
    initial_centroids = torch.tensor(kmeans_sklearn.cluster_centers_)
    # _, cluster_centers = kmeans(X=x, num_clusters=n_mem,  distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return initial_centroids