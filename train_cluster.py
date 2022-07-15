
########################################################################################################################
#
# train_cluster.py
# A python script to perform PCA transformation and clustering analysis using KMeans
#
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 07/07/2022
# Version: 1.0
#
# Usage:
#
# python train_cluster.py -i <Input csv data> -d <if drop first column>
#
##########################################################################################################################


import argparse
import os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def test_PCA(inData):
    # preprocessing
    # standardize variables
    ss = StandardScaler()
    if isinstance(inData, np.ndarray):
        X_std = ss.fit_transform(np.nan_to_num(inData))
    else:
        X_std = ss.fit_transform(np.nan_to_num(inData.values))
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
    pca = PCA(n_components=n_components).fit(X_std)
    X_pca = pca.transform(X_std)
    if isinstance(inData, np.ndarray):
        X_std = ss.fit(np.nan_to_num(inData))
    else:
        X_std = ss.fit(np.nan_to_num(inData.values))

    return X_std, pca, X_pca


def test_cluster(X_pca):
    n = list(range(3, 15))  # sequence of clusters to run
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    silhouette = []
    print('Processing clusters...')
    for i in tqdm(n):
        clust = KMeans(n_clusters=i)
        clust.fit(X_pca)
        cluster_labels = clust.predict(X_pca)
        silhouette.append(silhouette_score(X_pca, cluster_labels))

    clust = pd.concat([pd.DataFrame(n), pd.DataFrame(silhouette)], axis=1)
    clust.columns = ['clusters', 'silhouette']
    print(clust)
    plt.plot(n, silhouette, marker='o', alpha=0.75)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette index')
    plt.savefig('Silhouette.png', res=400)
    plt.show()
    # Stop!!! ask for the best n_Clusters to be used. Enter the number in the terminal
    n_clusters = int(input("Please enter your selected number of clusters: "))
    return KMeans(n_clusters=n_clusters).fit(X_pca)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inData', help='Input Data', type=str)
    parser.add_argument(
        '-d', '--drop', help='drop first column', action='store_true')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())

    # os.chdir('/home/javierlopatin/Documentos/temp/Siusun/')
    # inData = pd.read_csv('points_phenoshape.csv')
    inData = pd.read_csv(args['inData']).astype('float32')
    inData = inData.dropna()

    # drop undersirabe columns
    if args['drop']:
        inData.drop(inData.columns[[0]], axis=1, inplace=True)

    # test pca
    std, pca, X_pca = test_PCA(inData)
    # test clusters
    cluster = test_cluster(X_pca)

    # save pipeline with best models
    pipe = Pipeline([('std', std), ('pca', pca), ('cluster', cluster)])

    # save the model to disk
    filename = 'ClusterPipeline.sav'
    pickle.dump(pipe, open(filename, 'wb'))
