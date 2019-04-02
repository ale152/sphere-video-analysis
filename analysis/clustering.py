from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def location_clustering(features, loc_dist, loc_min, plot=True, savefig=False):
    """Cluster the features (typically location positions) using DBSCAN"""
    dbscan = DBSCAN(eps=loc_dist, min_samples=loc_min, metric='euclidean')
    pos_cluster = dbscan.fit(features)
    predicted_clusters = pos_cluster.labels_

    unique_clusters = np.unique(predicted_clusters)
    good_clusters = unique_clusters[unique_clusters >= 0]  # Remove outlier locations

    if plot:
        plt.figure(figsize=(5, 3))
        # Plot the clusters
        for clu in good_clusters:
            pt = features[predicted_clusters == clu, :]
            plt.plot(pt[:, 0], pt[:, 1], '.', label=('Location %d' % clu))

        # Plot outliers
        pt = features[predicted_clusters == -1, :]
        if pt.size > 0:
            plt.plot(pt[:, 0], pt[:, 1], 'x', label='Outliers')

        plt.legend()
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')
        plt.tight_layout()
        if savefig:
            plt.savefig('{}.png'.format(savefig))

    # Use just the position for clustering
    clusters = predicted_clusters
    n_clusters = len(good_clusters)
    return clusters, n_clusters


def gmm_clustering(features, gmm_n, pre_clusters=False, plot=True, savefig=False):
    """Cluster the features (typically bounding box shapes) using a Gaussian Mixture model. If pre_clusters are
    provided, the GNN is applied for each sub-cluster"""
    if not np.any(pre_clusters):
        pre_clusters = np.zeros((features.shape[0]))

    n_prec = len(np.unique(pre_clusters[pre_clusters >= 0]))
    clusters = np.zeros_like(pre_clusters)
    for ci in tqdm(range(n_prec)):
        sub_features = features[np.where(pre_clusters == ci)[0], :]
        gmm = GaussianMixture(n_components=gmm_n)
        gmm.fit(sub_features)
        prediction = gmm.predict(sub_features)
        # Sub-clusters will be different for each location cluster
        clusters[pre_clusters == ci] = prediction + ci * gmm_n

        if plot:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.title('Features for location {}'.format(ci))
            plt.scatter(sub_features[:, 0], sub_features[:, 1], c=sub_features[:, 2], cmap='jet')

            plt.subplot(1, 2, 2)
            for i in range(gmm_n):
                plt.plot(sub_features[prediction == i, 0], sub_features[prediction == i, 1], '.')
                plt.text(sub_features[prediction == i, 0].mean(), sub_features[prediction == i, 1].mean(),
                         'Sub {}'.format(i + ci * gmm_n))
            plt.title('Predicted identities')
            plt.legend(['Identity %d' % bf for bf in range(gmm_n)])

            if savefig:
                plt.savefig('{}_{}.png'.format(savefig, ci))

    n_clusters = len(np.unique(clusters))
    # Plot all the predicted identities for all the locations
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(features[:, 0], features[:, 1], c=features[:, 2], cmap='nipy_spectral')
        plt.xlabel('BB width')
        plt.xlabel('BB height')
        h = plt.colorbar()
        h.ax.set_ylabel('STS speed')
        for sub in range(n_clusters):
            sub_features = features[np.where(clusters == sub)[0], :]
            plt.subplot(1, 2, 2)
            plt.plot(sub_features[:, 0], sub_features[:, 1], '.')
            plt.text(sub_features[:, 0].mean(), sub_features[:, 1].mean(), 'Sub {}'.format(sub))
        if savefig:
                plt.savefig('{}_all.png'.format(savefig))

    return clusters, n_clusters
