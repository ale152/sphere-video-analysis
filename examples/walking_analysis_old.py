"""
walking_analysis.py
"""

import os
import json
from warnings import warn

import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from shutil import copyfile
# import skvideo.io
# from skvideo.io import vread


class WalkingAnalysis:
    def __init__(self, data_path, output_path, house_id=None):
        self.data_path = data_path
        self.output_path = output_path
        self.house_id = house_id

        # Load the list of bounding boxes
        self.box_files = self.search_box_files(data_path)

        # Initialise the arrays containing the results
        n_seq = len(self.box_files)
        self.store_profile = np.zeros((n_seq, 100))
        self.store_boxes = np.zeros((n_seq, 100, 11))
        self.all_box_loc = np.zeros((n_seq, 6))
        self.all_box_shape = np.zeros((n_seq, 2))
        self.all_speed = np.zeros((n_seq, 1))
        self.all_duration = np.zeros((n_seq, 1))
        self.all_snr = np.zeros((n_seq, 1))
        self.all_timestamp = np.zeros((n_seq, 1))
        self.clusters = np.zeros((n_seq, 1))
        self.n_clusters = 1

    @staticmethod
    def get_surgery_date(house_id):
        """Return the surgery date for some available houses"""
        if house_id == 4954:
            surgery_date = datetime.datetime(2017, 10, 21)
        elif house_id == 8145:
            surgery_date = datetime.datetime(2018, 3, 21)
        elif house_id == 3590:
            surgery_date = datetime.datetime(2018, 1, 12)
        elif house_id == 8731:
            surgery_date = datetime.datetime(2018, 2, 22)
        else:
            warn('Surgery date not available for {}'.format(house_id))
            surgery_date = None
        return surgery_date

    @staticmethod
    def search_box_files(data_path):
        """lorem ipsum"""
        box_list = []
        for folder, _, files in tqdm(os.walk(data_path)):
            # Loop over the files
            for file in files:
                if os.path.splitext(file)[1] == '.csv':
                    box_list.append(os.path.join(folder, file))



        # box_list = box_list[::100]


        return box_list

    def load_csv_from_path(self, csv_path):
        """Load the csv metadata related to a specific video. Accept extensions for .mp4, .csv and no extension,
        both for absolute and relative path"""
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(self.data_path, csv_path)

        # Deal with multiple possible extensions
        path_base, path_ext = os.path.splitext(csv_path)
        if path_ext == '.mp4':
            csv_path = path_base + '.csv'
        elif path_ext == '':
            csv_path = csv_path + '.csv'
        # Load the bounding box and format the timestamp
        box_data = np.loadtxt(csv_path, delimiter=',',
                              dtype=float, skiprows=1,
                              converters={0: lambda x: self.parse_time(x)})

        return box_data

    @staticmethod
    def parse_time(x):
        """Parse the time the way it was saved in the SPHERE database"""
        x = x.decode('utf-8')
        if len(x) == 26:
            y = datetime.datetime(*map(int, [x[:4], x[5:7], x[8:10], x[11:13], x[14:16], x[17:19], x[20:]]))
        else:
            y = datetime.datetime(*map(int, [x[:4], x[5:7], x[8:10], x[11:13], x[14:16], x[17:19], 0]))
        return y.timestamp()

    def speed_profile(self, top, time, smooth_kernel=11, smooth_poly=3):
        """lorem ipsum"""
        # First, filter the data and calculate the derivatives
        # cy_smooth = savgol_filter(y_box, smooth_kernel, smooth_poly, mode='interp')
        diff_x = savgol_filter(top[:, 0], smooth_kernel, smooth_poly, deriv=1, mode='interp')
        diff_z = savgol_filter(top[:, 2], smooth_kernel, smooth_poly, deriv=1, mode='interp')
        diff_t = savgol_filter(time, smooth_kernel, smooth_poly, deriv=1, mode='interp')

        speed_x = diff_x / diff_t
        speed_z = diff_z / diff_t

        speed = np.sqrt(np.square(speed_x) + np.square(speed_z))
        return speed

    def walking_speed(self, speed, speed_thr, duration_thr, smooth_kernel=11, smooth_poly=3):
        """lorem ipsum"""
        walking = (speed > speed_thr).astype('float')
        bounded = np.insert(walking, 0, 0)
        bounded = np.append(bounded, 0)
        right = np.where(np.diff(bounded) < 0)[0]
        left = np.where(np.diff(bounded) > 0)[0]
        duration = right - left
        if len(duration) > 0 and duration.max()/100 > duration_thr:
            pos = np.argmax(duration)
            walking_speed = np.mean(speed[left[pos]:right[pos]])
            return walking_speed, duration.max(), left[pos]
        else:
            return np.nan, np.nan, 0

    @staticmethod
    def find_peaks(y):
        """Find peak locations in the vector y. Peaks are defined as points preceded and followed by a lower value"""
        diff_y = np.diff(y)
        sign_diff = np.sign(diff_y)
        diff_sign_diff = np.diff(sign_diff)
        pks = np.where(diff_sign_diff == -2)[0] + 1
        return pks, y[pks]

    @staticmethod
    def signal_to_noise(pks_val):
        """Return the signal to noise of a vector containing peaks. The snr is defined as ratio between highest and
        second highest peak"""
        if len(pks_val) > 1:
            tmp = pks_val.copy()
            max_pos = tmp.argmax()
            max_val = tmp[max_pos]
            tmp[max_pos] = -np.inf
            snr = max_val / tmp.max()
            return snr
        else:
            warn('Signal to noise requires at least two elements to be computed, while {} were provided. '
                 'Returning NaN'.format(len(pks_val)))
            return np.nan

    def location_clustering(self, features, loc_dist, loc_min, plot=True):
        """The first clustering must be done using the position of the bounding box in 3D"""
        dbscan = DBSCAN(eps=loc_dist, min_samples=loc_min, metric='euclidean')
        pos_cluster = dbscan.fit(features)
        predicted_locations = pos_cluster.labels_

        all_locations = np.unique(predicted_locations)
        good_locations = all_locations[all_locations >= 0]  # Remove outlier locations

        if plot:
            plt.figure(figsize=(5, 3))
            # Plot the clusters
            for clu in good_locations:
                pt = features[predicted_locations == clu, :]
                plt.plot(pt[:, 0], pt[:, 1], '.', label=('Location %d' % clu))

            # Plot outliers
            pt = features[predicted_locations == -1, :]
            if pt.size > 0:
                plt.plot(pt[:, 0], pt[:, 1], 'x', label='Outliers')

            plt.legend()
            plt.xlabel('x position (m)')
            plt.ylabel('y position (m)')
            plt.tight_layout()
            plt.savefig('location_clusters.png')

        # Use just the position for clustering
        self.clusters = predicted_locations
        self.n_clusters = len(good_locations)

    def participants_gmm(self, features, gmm_n, plot=True):
        """Start from the location clusters and further discriminate between "gmm_n" participants using a Gaussian
        Mixture"""
        sub_clusters = np.zeros_like(self.clusters)
        for clu in tqdm(range(self.n_clusters)):
            sub_features = features[np.where(self.clusters == clu)[0], :]
            gmm = GaussianMixture(n_components=gmm_n)
            gmm.fit(sub_features)
            prediction = gmm.predict(sub_features)
            # Sub-clusters will be different for each location cluster
            sub_clusters[self.clusters == clu] = prediction + clu * gmm_n

            if plot:
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.title('Features for location {}'.format(self.clusters[clu]))
                plt.scatter(sub_features[:, 0], sub_features[:, 1], c=sub_features[:, 2], cmap='jet')

                plt.subplot(1, 2, 2)
                for i in range(gmm_n):
                    plt.plot(sub_features[prediction == i, 0], sub_features[prediction == i, 1], '.')
                    plt.text(sub_features[prediction == i, 0].mean(), sub_features[prediction == i, 1].mean(),
                             'Sub {}'.format(i + clu * gmm_n))
                plt.title('Predicted identities')
                plt.legend(['Identity %d' % bf for bf in range(gmm_n)])

        n_detected_subjects = len(np.unique(sub_clusters))
        # Plot all the predicted identities for all the locations
        if plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(features[:, 0], features[:, 1], c=features[:, 2], cmap='nipy_spectral')
            plt.xlabel('BB width')
            plt.xlabel('BB height')
            h = plt.colorbar()
            h.ax.set_ylabel('STS speed')
            for sub in range(n_detected_subjects):
                sub_features = features[np.where(sub_clusters == sub)[0], :]
                plt.subplot(1, 2, 2)
                plt.plot(sub_features[:, 0], sub_features[:, 1], '.')
                plt.text(sub_features[:, 0].mean(), sub_features[:, 1].mean(), 'Sub {}'.format(sub))

        self.clusters = sub_clusters
        self.n_clusters = n_detected_subjects

    def load_data(self):
        # Loop over all the stand up transitions
        for vi, vid in enumerate(tqdm(self.box_files)):
            # Load the box data and store the timestamp
            box_data = self.load_csv_from_path(vid)
            self.store_boxes[vi, ...] = box_data

            # Calculate the speed profile
            top = box_data[:, 5:8] / 1000
            time = box_data[:, 0]
            self.store_profile[vi, :] = self.speed_profile(top, time)

    def evaluate(self, speed_thr, duration_thr):
        n_seq = len(self.box_files)
        self.all_box_loc = np.zeros((n_seq, 6))
        self.all_box_shape = np.zeros((n_seq, 2))
        self.all_speed = np.zeros((n_seq, 1))
        self.all_duration = np.zeros((n_seq, 1))
        self.all_snr = np.zeros((n_seq, 1))
        self.all_timestamp = np.zeros((n_seq, 1))
        self.clusters = np.zeros((n_seq, 1))
        self.n_clusters = 1

        for vi in range(len(self.box_files)):
            profile = self.store_profile[vi, :]
            walking_speed, duration, pos = self.walking_speed(profile, speed_thr, duration_thr)
            self.all_duration[vi] = duration
            self.all_speed[vi] = walking_speed

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
        self.all_speed = self.all_speed[good, :]
        self.all_duration = self.all_duration[good, :]
        self.all_snr = self.all_snr[good]
        self.all_timestamp = self.all_timestamp[good]
        self.clusters = self.clusters[good]

    def cluster(self, by_location, by_participant, loc_dist=0.2, loc_min=10, gmm_n=2):
        # The first clustering must be done using the position of the bounding box in 3D
        self.clusters[:] = 0
        self.n_clusters = 1
        if by_location:
            features = (self.all_box_loc[:, :3] + self.all_box_loc[:, 3:])/2
            self.location_clustering(features, loc_dist, loc_min)

        # The second clustering is based on the 3D bounding box shape and STS speed, for each location cluster
        if by_participant:
            features = np.hstack((self.all_box_shape, self.all_speed))
            self.participants_gmm(features, gmm_n)


if __name__ == '__main__':
    house_id = 9665
    data_path = r'G:\STS_sequences_overlap\{}\Videos'.format(house_id)
    output_path = r'G:\STS_results\walking_speed'
    # min_dist = 0.2#0.2#0.25  # In meters. The standard sofa cushion size is 18 inches, about 0.5 m
    # min_n_sit = 10#50#10  # Min number of times someone must sit there to be considered a sitting location
    target = 'stand up'
    sts = WalkingAnalysis(data_path, output_path, house_id)

    sts.load_data()
    sts.evaluate(speed_thr=0.1, duration_thr=0.5)
    # sts.filter(snr_thr=2)
    sts.cluster(by_location=True, by_participant=False, loc_dist=0.5, loc_min=10, gmm_n=2)

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
    period = 'W'
    # Convert the file name into a date object
    surgery_date = sts.get_surgery_date(house_id)
    figures = []
    for cluster_i in range(sts.n_clusters):
        select = sts.clusters == cluster_i
        sub_speed = sts.all_speed[select]
        sub_date = [datetime.datetime.fromtimestamp(bf) for bf in sts.all_timestamp[select]]

        df = pd.DataFrame(data=sub_speed, index=sub_date)
        agg = df.resample(period)
        res = agg.mean()

        fig = plt.figure(figsize=(5, 3))
        figures.append(fig)
        plt.errorbar(res.index, res.get_values(), agg.std().get_values())
        if surgery_date:
            plt.axvline(surgery_date, linewidth=2, color='k', label='Surgery day')

        plt.title('Cluster {}'.format(cluster_i))

    # Set the same limits for all the figures
    xlim = [np.min(np.array([bf.gca().get_xlim() for bf in figures]), axis=0)[0],
            np.max(np.array([bf.gca().get_xlim() for bf in figures]), axis=0)[1]]
    ylim = [np.min(np.array([bf.gca().get_ylim() for bf in figures]), axis=0)[0],
            np.max(np.array([bf.gca().get_ylim() for bf in figures]), axis=0)[1]]
    for fig in figures:
        fig.gca().set_xlim(xlim)
        fig.gca().set_ylim(ylim)



# if __name__ == '__main__':
#     main()
