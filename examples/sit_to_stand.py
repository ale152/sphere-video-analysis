"""
sit_to_stand.py defines the framework SitToStand to the analysis of the sit-to-stand transitions.
"""

import os
from warnings import warn

import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

from sphere.utils import get_surgery_date
from sphere.data_loader import labels_loader as dl
from sphere.analysis.clustering import location_clustering, gmm_clustering
from sphere.analysis.metrics import speed_of_ascent
from sphere.analysis.plots import trend_plot
from sphere.utils.animate_utils import play_sts_sequence, play_bson_silhouette_and_wearable

class SitToStand:
    def __init__(self, data_path, labels_path, output_path, target='stand up', house_id=None, npz_thresh=0.0):
        self.data_path = data_path
        self.labels_path = labels_path
        self.output_path = output_path
        self.target = target
        self.house_id = house_id

        # Load the annotations from the json file
        self.meta_list, \
        self.data_mode, \
        self.zips_list, \
        self.in_which_zip = dl.load_labels_from_file(labels_path, data_path, target, thresh=npz_thresh)

        # Initialise the arrays containing the results
        n_sts = len(self.meta_list)
        self.all_box_loc = np.zeros((n_sts, 6))
        self.all_box_shape = np.zeros((n_sts, 2))
        self.metrics = {}
        self.metrics['speed'] = np.zeros((n_sts))
        self.metrics['snr'] = np.zeros((n_sts))
        self.all_timestamp = np.zeros((n_sts))
        self.clusters = np.zeros((n_sts))
        self.n_clusters = 1

    def evaluate(self):
        # Loop over all the stand up transitions
        n_lab = len(self.meta_list)
        for vi in tqdm(range(n_lab)):
            # Load the box data and store the timestamp
            box_data = dl.load_boxes(vi, self.meta_list, self.data_mode, self.data_path,
                                     self.zips_list, self.in_which_zip)
            self.all_timestamp[vi] = box_data[0, 0]

            # Store the maximum width and eight of the boxes
            max_shape = np.max(np.abs(box_data[:, 5:7] - box_data[:, 8:10]) / 1000, axis=0)
            self.all_box_shape[vi, :] = max_shape

            # Save the first or last n coordinates for the clustering
            if self.target == 'stand up':
                self.all_box_loc[vi, :] = box_data[0, -6:] / 1000
            else:
                self.all_box_loc[vi, :] = box_data[-1, -6:] / 1000

            # Calculate the STS speed
            cy = box_data[:, 6] / 1000  # Use the y coordinate of the upper edge of the 3D bounding box
            time = box_data[:, 0]
            speed, snr = speed_of_ascent(cy, time, self.target)
            self.metrics['speed'][vi] = speed
            self.metrics['snr'][vi] = snr

    def filter(self, snr_thr=False):
        if snr_thr:
            good = np.where(self.metrics['snr'] >= snr_thr)[0]
            self.all_box_loc = self.all_box_loc[good, :]
            self.all_box_shape = self.all_box_shape[good, :]
            self.metrics['speed'] = self.metrics['speed'][good, :]
            self.metrics['snr'] = self.metrics['snr'][good]
            self.all_timestamp = self.all_timestamp[good]
            self.clusters = self.clusters[good]

    def cluster(self, by_location, by_participant, loc_dist=0.2, loc_min=10, gmm_n=2):
        # Reset the clusters
        n_sts = len(self.meta_list)
        self.clusters = np.zeros((n_sts, 1))
        self.n_clusters = 1

        # The first clustering must be done using the position of the bounding box in 3D
        if by_location:
            features = (self.all_box_loc[:, :3] + self.all_box_loc[:, 3:])/2
            self.clusters, self.n_clusters = location_clustering(features, loc_dist, loc_min)

        # The second clustering is based on the 3D bounding box shape and STS speed, for each location cluster
        if by_participant:
            features = np.hstack((self.all_box_shape, self.metrics['speed'][:, None]))
            self.clusters, self.n_clusters = gmm_clustering(features, gmm_n, pre_clusters=self.clusters)


if __name__ == '__main__':
    # house_id = 3590
    # data_path = r'I:\STS_zipped_overlap_50\{}'.format(house_id)
    # labels_path = r'I:\STS_zipped_overlap_50\automatic_labels_{}.npz'.format(house_id)
    house_id = 8145
    data_path = r'G:\STS_sequences_overlap\{}'.format(house_id)
    # labels_path = r'G:\STS_sequences_overlap\{}\labels_{}_livingroom_revised.json'.format(house_id, house_id)
    labels_path = r"C:\Users\am14795\Dropbox\documenti\SPHERE\keras_play\STS_classifier\automatic_labels_%d.json" % \
                  house_id
    output_path = r'G:\STS_results\transition_derivative'
    # min_dist = 0.2#0.2#0.25  # In meters. The standard sofa cushion size is 18 inches, about 0.5 m
    # min_n_sit = 10#50#10  # Min number of times someone must sit there to be considered a sitting location
    target = 'stand up'
    sts = SitToStand(data_path, labels_path, output_path, target, house_id)

    sts.evaluate()
    # sts.filter(snr_thr=2)
    sts.cluster(by_location=True, by_participant=False)

    # # %% Clock plot
    # for cluster_i in range(sts.n_clusters):`
    #     select = sts.clusters == cluster_i
    #     sub_speed = sts.all_speed[select]
    #     sub_date = [datetime.datetime.fromtimestamp(bf) for bf in sts.all_timestamp[select]]
    #
    #     df = pd.DataFrame(data=sub_speed, index=sub_date)
    #     agg = df.groupby(df.index.hour)
    #
    #     plt.figure(figsize=(5, 3))
    #     plt.subplot(121)
    #     plt.bar(agg.count().index, agg.count().get_values()[:,0])
    #     plt.subplot(122)
    #     plt.bar(agg.mean().index, agg.mean().get_values()[:,0])
    #     plt.title('Cluster {}'.format(cluster_i))

    # play_sts_sequence(sts.all_speed.argsort()[-1], sts)

    surgery_date = get_surgery_date(house_id)
    trend_plot(sts, 'speed', 7, surgery_date, linear_trend=True)

    # # Set the same limits for all the figures
    # xlim = [np.min(np.array([bf.gca().get_xlim() for bf in figures]), axis=0)[0],
    #         np.max(np.array([bf.gca().get_xlim() for bf in figures]), axis=0)[1]]
    # ylim = [np.min(np.array([bf.gca().get_ylim() for bf in figures]), axis=0)[0],
    #         np.max(np.array([bf.gca().get_ylim() for bf in figures]), axis=0)[1]]
    # for fig in figures:
    #     fig.gca().set_xlim(xlim)
    #     fig.gca().set_ylim(ylim)

    # %%
    # start = datetime.datetime.fromtimestamp(sts.all_timestamp[30] - 10)
    # end = datetime.datetime.fromtimestamp(sts.all_timestamp[30] + 50)
    # play_bson_silhouette_and_wearable(r'G:\HouseData-Sphere\4954\backups', start, end, 'liv', export_video=True)

# if __name__ == '__main__':
#     main()
