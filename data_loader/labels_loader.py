"""
This module allows loading SPHERE data. The data can be stored in two different formats:
* Video files
    House data is divided in sub-folders by day. Each day folder contains a .mp4 file for each video clip and a .csv
    file for the meta-data (bounding boxes and timestamp).
    This format is used for labelling and comes with a .json file containing the annotations
* Zip archive
    House data is divided in zip "chunks", in the format of "chunk_%d_meta.zip" and "chunk_%d_video.zip". Each zip
    file contains up to 1000 video sequences and is organized by sub-folder per day.
    This video format is used for training/testing the network. If labelled, the zip names will contain the label as
    well ("sit_down_video.zip", "stand_up_meta.zip", "negative_chunk_%d_video.zip", ...)
"""
import os
import json
import zipfile
from io import StringIO, BytesIO

import numpy as np
from PIL import Image
from tqdm import tqdm
from skvideo.io import vread

from sphere.utils.date_utils import parse_time


def load_labels_from_file(labels_path, data_path, target, thresh=0.0):
    """Open the labels file from a path. The labels can either be a json format or a numpy archive. If numpy archive
    is provided, the function look for the zip files where the annotated files are stored.
    Labels are the path of the video clips for a specific action (stand up or sit down)"""
    ext = os.path.splitext(labels_path)[1]
    data_mode = ext[1:]
    if data_mode == 'json':
        label_files = _labels_from_json(labels_path, target)
        zips_list, in_which_zip = [], []
    elif data_mode == 'npz':
        label_files = _labels_from_numpy(labels_path, target, thresh=thresh)
        zips_list, in_which_zip = _generate_zip_mapping_from_labels(label_files, data_path)
    else:
        raise Exception('The labels path must be a json file or a numpy archive')

    return label_files, data_mode, zips_list, in_which_zip


def load_all_meta_files(data_path):
    """Search meta files containing information for the bounding boxes"""
    print('Loading data from {}'.format(data_path))
    data_mode = _detect_data_mode(data_path)
    if data_mode == 'json':
        meta_list = _scan_folders_by_ext(data_path, '.csv')
        zips_list = []
        in_which_zip = []
    elif data_mode == 'npz':
        meta_list, zips_list, in_which_zip = _generate_zip_mapping_from_folder(data_path)
    else:
        raise Exception('Data mode must be either json or npz')

    return meta_list, data_mode, zips_list, in_which_zip


def load_boxes(idx, meta_files, data_mode, data_path, zips_list, in_which_zip):
    """Load the csv metadata related to a specific video. Accept extensions for .mp4, .csv and no extension,
    both for absolute and relative path"""
    meta_path = meta_files[idx]
    if data_mode == 'json':
        if not os.path.isabs(meta_path):
            meta_path = os.path.join(data_path, meta_path)
    # Deal with multiple possible extensions
    path_base, path_ext = os.path.splitext(meta_path)
    if path_ext == '.mp4':
        meta_path = path_base + '.csv'
    elif path_ext == '':
        meta_path = meta_path + '.csv'

    if data_mode == 'npz':
        right_zip = zips_list[in_which_zip[idx]]
        zip_path = os.path.join(data_path, right_zip)
        with zipfile.ZipFile(zip_path) as my_zip:
            # meta_path = _convert_csv_path_to_zip(meta_path)
            meta_path = StringIO(my_zip.read(meta_path).decode('utf8'))

    # Load the bounding box and format the timestamp
    box_data = _load_boxes_np(meta_path)
    return box_data


def load_frames(idx, video_files, data_mode, data_path, zips_list, in_which_zip):
    """Load the video frames"""
    if data_mode == 'json':
        vid_path = video_files[idx]
        if not os.path.isabs(vid_path):
            vid_path = os.path.join(data_path, vid_path)
        # Deal with multiple possible extensions
        path_base, path_ext = os.path.splitext(vid_path)
        if path_ext == '.csv':
            vid_path = path_base + '.mp4'
        elif path_ext == '':
            vid_path = vid_path + '.mp4'
        frames = vread(vid_path)
    elif data_mode == 'npz':
        vid_path = os.path.dirname(video_files[idx])
        right_zip = zips_list[in_which_zip[idx]]
        zip_path = os.path.join(data_path, right_zip)  # type: str
        zip_path = zip_path.split('_meta.zip')[0] + '_video.zip'
        with zipfile.ZipFile(zip_path) as my_zip:
            frames = np.zeros((100, 100, 100, 3), dtype=np.uint8)
            for i in range(100):
                img_path = os.path.join(vid_path, '%04d.jpg' % i)
                img_path = os.path.normpath(img_path).replace('\\', '/')
                file = my_zip.read(img_path)
                image = np.array(Image.open(BytesIO(file), ))
                image = np.dstack((image, image, image))
                frames[i, ...] = image
    else:
        raise Exception('Data mode must be either json or npz')

    return frames


def _load_boxes_np(box_path):
    """Load the bounding boxes"""
    box_data = np.loadtxt(box_path, delimiter=',', dtype=float, skiprows=1, converters={0: lambda x: parse_time(x)})
    # /!\ The data in the database might not be sorted, sort it now /!\
    order = np.argsort(box_data[:, 0])
    box_data = box_data[order, ...]
    return box_data


def _labels_from_json(labels_path, target):
    """Load the labels from a json file"""
    with open(labels_path, 'r') as labels_path:
        label_files = json.load(labels_path)
        label_files = [bf['video'] for bf in label_files if bf['label'] == target]
    return label_files


def _generate_zip_mapping_from_labels(label_files, zip_path):
    """Given a list of annotations (video paths), returns the list of zipfiles and the location of each csv file in the
    zips"""
    # First, find all the zipped files and their content
    n_labels = len(label_files)
    zips_list = [bf for bf in os.listdir(zip_path) if bf.endswith('meta.zip')]
    in_which_zip = [None for _ in range(n_labels)]
    print('Generate zip mapping from labels')
    for zi, zipf in enumerate(tqdm(zips_list)):
        with zipfile.ZipFile(os.path.join(zip_path, zipf)) as myzip:
            myzip_namelist = myzip.namelist()
            for di, dat in enumerate(label_files):  # type: (int, str)
                if dat in myzip_namelist:
                    in_which_zip[di] = zi

    return zips_list, in_which_zip


def _generate_zip_mapping_from_folder(zip_path):
    """Given a folder with _meta zip files, returns the list of zipfiles and the location of each csv file in the
    zips"""
    # First, find all the zipped files and their content
    zips_list = [bf for bf in os.listdir(zip_path) if bf.endswith('meta.zip')]
    in_which_zip = []
    meta_list = []
    print('Generate zip mapping from folder')
    for zi, zipf in enumerate(tqdm(zips_list)):
        with zipfile.ZipFile(os.path.join(zip_path, zipf)) as myzip:
            myzip_namelist = myzip.namelist()
            meta_list.extend(myzip_namelist)
            in_which_zip.extend([zi for _ in range(len(myzip_namelist))])

    return meta_list, zips_list, in_which_zip


def _convert_csv_path_to_zip(csv_path):
    """The bounding box information is stored in a different way according to the data mode. The in 'json' mode the
    csv path is simply the path to the json file. In 'npz' mode the csv file is stored in a zip file. This function
    converts the csv path from json to npz mode."""
    f1 = os.path.dirname(csv_path)
    f2 = os.path.basename(csv_path)
    f3 = os.path.basename(csv_path) + '.csv'
    new_name = os.path.join(f1, f2, f3)
    new_name = os.path.normpath(new_name).replace('\\', '/')
    return new_name


def _convert_mp4_path_to_zip(vid_path):
    """dasdsadasda."""
    f1 = os.path.dirname(vid_path)
    f2 = os.path.basename(vid_path)
    f3 = os.path.basename(vid_path) + '.mp4'
    new_name = os.path.join(f1, f2, f3)
    new_name = os.path.normpath(new_name).replace('\\', '/')
    return new_name


def _labels_from_numpy(labels_path, target, thresh=0.0):
    """Load the labels from a numpy archive (typically the output of the network)"""
    with np.load(labels_path) as bf:
        all_results = bf['all_results']
        all_predictions = bf['all_predictions']
        all_videos = bf['all_videos']
        if thresh:
            prob = all_results.max(axis=1)
            all_predictions = all_predictions[prob >= thresh]
            all_videos = all_videos[prob >= thresh]
    # Filter the labels by the specified target
    unique_labels = ('sit down', 'stand up', 'other')  # As defined in BatchGenerator
    target_ind = unique_labels.index(target)
    n_all = len(all_predictions)
    label_files = [_convert_csv_path_to_zip(all_videos[bf]) for bf in range(n_all) if all_predictions[bf] == target_ind]
    return label_files


def _detect_data_mode(data_path):
    test_1 = [bf for bf in os.listdir(data_path) if bf.endswith('.zip')] if os.path.exists(data_path) else []
    bf = os.path.join(data_path, 'Videos')
    test_2 = [bf for bf in os.listdir(bf) if bf.endswith('.zip')] if os.path.exists(bf) else []
    bf = os.path.join(data_path, 'Videos', 'train')
    test_3 = [bf for bf in os.listdir(bf) if bf.endswith('.zip')] if os.path.exists(bf) else []
    if len(test_1) or len(test_2) or len(test_3):
        return 'npz'
    else:
        return 'json'


def _scan_folders_by_ext(folder, ext):
    """Scan a folder and sub-folders and return all the files that match a specific extension"""
    box_list = []
    print('Scanning folder {} by extension {}'.format(folder, ext))
    for sub_folder, _, files in tqdm(os.walk(folder)):
        # Loop over the files
        for file in files:
            if os.path.splitext(file)[1] == ext:
                box_list.append(os.path.join(sub_folder, file))

    return box_list
