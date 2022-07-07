
########################################################################################################################
#
# train_cluster.py
# A python script to perform PCA transformation and clustering analysis using AgglomerativeClustering
#
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 07/07/2022
# Version: 1.0
#
# Usage:
#
# python train_cluster.py -i <Input csv data>
#
##########################################################################################################################


import argparse
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def test_PCA(inData):
    # preprocessing
    # standardize variables
    ss = StandardScaler()
    X_std = ss.fit_transform(np.nan_to_num(inData))
    # PCA
    pca = PCA()  # PCA with 3 primary components
    # fit and transform both PCA models
    X_pca = pca.fit_transform(X_std)
    variance = pca.explained_variance_ratio_
    print('Cumulative PCA variance: ')
    comp = pd.concat([pd.DataFrame(np.linspace(
        1, inData.shape[1], inData.shape[1])), pd.DataFrame(np.cumsum(variance))], axis=1)
    comp.columns = ['n_comp', 'cumsum']
    print(comp)
    # Stop!!! ask for the best n_Clusters to be used. Enter the number in the terminal
    n_components = int(
        input("Please enter your selected number of components: "))
    return PCA(n_components=n_components).fit(X_std)


def test_cluster(X_pca):
    n = list(range(3, 15))  # sequence of clusters to run
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    silhouette = []
    print('Processing clusters...')
    for i in tqdm(n):
        clust = AgglomerativeClustering(n_clusters=i)
        clust.fit(X_pca)
        cluster_labels = clust.fit_predict(X_pca)
        silhouette.append(silhouette_score(X_pca, cluster_labels))

    clust = pd.concat([pd.DataFrame(n), pd.DataFrame(silhouette)], axis=1)
    clust.columns = ['clusters', 'silhouette']
    print(clust)
    plt.plot(n, silhouette, marker='o', alpha=0.75)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette index')
    plt.savefig('Silhouette.png', res=400))
    plt.show()
    # Stop!!! ask for the best n_Clusters to be used. Enter the number in the terminal
    n_clusters = int(input("Please enter your selected number of clusters: "))
    return AgglomerativeClustering(n_clusters=n_clusters).fit(X_pca)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inData', help='Input Data', type=str)
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())

    inData = pd.read_csv(args['inData'])
    inData = np.nan_to_num(inData)

    # test pca
    pca = test_PCA(inData)
    X_pca = pca.transform(inData)
    # test clusters
    cluster = test_cluster(X_pca)

    # save pipeline with best models
    pipe = Pipeline([('pca', pca), ('cluster', cluster)])
    joblib.dump(pipe, 'ClusterPipeline.pkl')
