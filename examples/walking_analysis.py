"""
walking_analysis.py
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from sphere.utils import get_surgery_date
import sphere.data_loader.labels_loader as dl
from sphere.analysis.clustering import location_clustering, gmm_clustering
from sphere.analysis.metrics import horizontal_velocity, walking_speed
from sphere.analysis.plots import trend_plot
from sphere.utils.animate_utils import play_walking_sequence, play_bson_silhouette, play_bson_silhouette_and_wearable

class WalkingAnalysis:
    def __init__(self, data_path, output_path, house_id=None):
        self.data_path = data_path
        self.output_path = output_path
        self.house_id = house_id

        # Load the list of bounding boxes
        self.meta_list,\
        self.data_mode, \
        self.zips_list, \
        self.in_which_zip = dl.load_all_meta_files(data_path)

        # Load all the metadata and store them
        n_seq = len(self.meta_list)
        self.store_profile = np.zeros((n_seq, 100))
        self.store_boxes = np.zeros((n_seq, 100, 11))
        self.load_data()

        # Initialise empty variables
        self.all_box_loc = np.zeros((n_seq, 6))
        self.all_box_shape = np.zeros((n_seq, 2))
        self.all_speed = np.zeros((n_seq))
        self.all_duration = np.zeros((n_seq))
        self.all_snr = np.zeros((n_seq))
        self.all_timestamp = np.zeros((n_seq))
        self.all_good = np.zeros((n_seq))
        self.clusters = np.zeros((n_seq))
        self.n_clusters = 1

    def load_data(self):
        tmp_name = os.path.join(os.path.dirname(self.data_path),
                                'tmp_walking_' + os.path.basename(self.data_path) + '.npz')
        if not os.path.exists(tmp_name):
            # Loop over all the stand up transitions
            n_seq = len(self.meta_list)
            for vi in tqdm(range(n_seq)):
                # Load the box data and store the timestamp
                box_data = dl.load_boxes(vi, self.meta_list, self.data_mode, self.data_path,
                                         self.zips_list, self.in_which_zip)
                self.store_boxes[vi, ...] = box_data

                # Calculate the speed profile
                top = box_data[:, 5:8] / 1000
                time = box_data[:, 0]
                self.store_profile[vi, :] = horizontal_velocity(top, time)

            np.savez(tmp_name, store_boxes=self.store_boxes, store_profile=self.store_profile)
        else:
            print('Data loaded from temporary file {}'.format(tmp_name))
            with np.load(tmp_name) as bf:
                self.store_boxes = bf['store_boxes']
                self.store_profile = bf['store_profile']

    def histogram(self):
        n_meta = len(self.meta_list)
        max_speed = np.zeros((n_meta))
        for vi in tqdm(range(n_meta)):
            profile = self.store_profile[vi, :]
            max_speed[vi] = profile.max()


        plt.hist(max_speed, 300)
        plt.yscale('log')

    def evaluate(self, speed_thr, duration_thr):
        n_seq = len(self.meta_list)
        self.all_box_loc = np.zeros((n_seq, 6))
        self.all_box_shape = np.zeros((n_seq, 2))
        self.all_speed = np.zeros((n_seq))
        self.all_duration = np.zeros((n_seq))
        self.all_snr = np.zeros((n_seq))
        self.all_timestamp = np.zeros((n_seq))
        self.all_good = np.zeros((n_seq))
        self.clusters = np.zeros((n_seq))
        self.n_clusters = 1

        for vi in tqdm(range(len(self.meta_list))):
            profile = self.store_profile[vi, :]
            speed, duration, pos = walking_speed(profile, speed_thr, duration_thr)
            self.all_duration[vi] = duration
            self.all_speed[vi] = speed

            box_data = self.store_boxes[vi, ...]
            self.all_timestamp[vi] = box_data[0, 0]

            # Store the maximum width and eight of the boxes
            max_shape = np.max(np.abs(box_data[:, 5:7] - box_data[:, 8:10]) / 1000, axis=0)
            self.all_box_shape[vi, :] = max_shape
            self.all_box_loc[vi] = box_data[50, -6:] / 1000

        # Remove the non valid walking speeds
        good = np.where(np.logical_not(np.isnan(self.all_speed)))[0]
        self.all_box_loc = self.all_box_loc[good, :]
        self.all_box_shape = self.all_box_shape[good, :]
        self.all_speed = self.all_speed[good]
        self.all_duration = self.all_duration[good]
        self.all_snr = self.all_snr[good]
        self.all_timestamp = self.all_timestamp[good]
        self.all_good = good
        self.clusters = self.clusters[good]
        print('Total number of points: {}'.format(good.sum()))

    def cluster(self, by_location, by_participant, loc_dist=0.2, loc_min=10, gmm_n=2):
        # Reset the clusters
        n_meta = len(self.all_speed)
        self.clusters = np.zeros((n_meta, 1))
        self.n_clusters = 1

        # The first clustering must be done using the position of the bounding box in 3D
        if by_location:
            features = (self.all_box_loc[:, :3] + self.all_box_loc[:, 3:])/2
            self.clusters, self.n_clusters = location_clustering(features, loc_dist, loc_min)

        # The second clustering is based on the 3D bounding box shape and STS speed, for each location cluster
        if by_participant:
            features = np.hstack((self.all_box_shape, self.all_speed))
            self.clusters, self.n_clusters = gmm_clustering(features, gmm_n, pre_clusters=self.clusters)


if __name__ == '__main__':
    house_id = 4954
    data_path = r'G:\STS_zipped_overlap_50\{}\liv'.format(house_id)
    # house_id = 4954
    # data_path = r'G:\STS_sequences_overlap\{}\Videos'.format(house_id)
    output_path = r'J:\STS_results\walking_speed'
    # min_dist = 0.2#0.2#0.25  # In meters. The standard sofa cushion size is 18 inches, about 0.5 m
    # min_n_sit = 10#50#10  # Min number of times someone must sit there to be considered a sitting location
    wks = WalkingAnalysis(data_path, output_path, house_id)

    wks.evaluate(speed_thr=0.1, duration_thr=0.5)
    # sts.filter(snr_thr=2)
    wks.cluster(by_location=True, by_participant=False, loc_dist=0.5, loc_min=10, gmm_n=3)

    # # %% Clock plot
    # for cluster_i in range(sts.n_clusters):
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

    # %% Weekly plot
    surgery_date = get_surgery_date(house_id)
    trend_plot(wks, 7, surgery_date)

    # %% Investigate full silhouette
    wid = 10
    play_walking_sequence(wks.all_good[wid], wks)

    start = datetime.datetime.fromtimestamp(wks.all_timestamp[wid])
    end = datetime.datetime.fromtimestamp(wks.all_timestamp[wid] + 10)
    play_bson_silhouette(r'G:\HouseData-Sphere\4954\backups', start, end, 'liv')

    start = datetime.datetime.fromtimestamp(wks.all_timestamp[wid] - 10)
    end = datetime.datetime.fromtimestamp(wks.all_timestamp[wid] + 50)
    play_bson_silhouette_and_wearable(r'G:\HouseData-Sphere\4954\backups', start, end, 'liv', export_video=True)
