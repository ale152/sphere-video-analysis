import os
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from scipy.signal import savgol_filter
from tqdm import tqdm
from skvideo.io import vwrite
from scipy import ndimage

from sphere.data_loader.labels_loader import load_boxes, load_frames
from sphere.data_loader.mongo_loader import BsonFile
from sphere.analysis.metrics import speed_of_ascent, walking_speed, horizontal_velocity
from sphere.utils import savgol_deriv
from sphere.utils.image_utils import b64_image_to_numpy
from sphere.utils.mongo_utils import read_room_uid


def play_walking_mosaic(id_list, obj, export_video=False):
    format = 16/9
    n_id = len(id_list)
    n_y = int(np.ceil(np.sqrt(n_id / format)))
    n_x = int(np.ceil(np.sqrt(n_id * format)))
    fig, ax = plt.subplots(n_y, n_x * 2)

    # Pre-load all frames and bounding boxes
    frames = []
    boxes = []
    time = []
    velocity_profile = []
    for id in id_list:
        tmp_frames = load_frames(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)
        tmp_boxes = load_boxes(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)
        tmp_velocity = horizontal_velocity(tmp_boxes[:, 5:8] / 1000, tmp_boxes[:, 0])
        frames.append(tmp_frames)
        boxes.append(tmp_boxes)
        time.append(tmp_boxes[:, 0])
        velocity_profile.append(tmp_velocity)

    # Initialise the mosaic
    imshows = []
    cursors = []
    for i, (yi, xi) in enumerate(product(range(n_y), range(n_x))):
        if i == len(time):
            break
        print('{}) {} {}'.format(i, xi, yi))

        imshows.append(ax[yi, xi*2].imshow(frames[i][0, ...]))
        ax[yi, xi*2].set_xticks([])
        ax[yi, xi*2].set_yticks([])

        ax[yi, xi*2-1].plot(time[i], velocity_profile[i])
        ax[yi, xi*2-1].set_xticks([])
        ax[yi, xi*2-1].set_yticks([])
        cursors.append(ax[yi, xi*2-1].axvline(time[i][0], color='k'))

    # Animate
    for ti in range(100):
        print(ti)
        for i, (yi, xi) in enumerate(product(range(n_y), range(n_x))):
            if i == len(time):
                break
            imshows[i].set_data(frames[i][ti, ...])
            cursors[i].set_xdata(time[i][ti])
            plt.pause(0.0001)


def play_walking_sequence(id, obj, export_video=False):
    frames = load_frames(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)
    boxes = load_boxes(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)

    # Calculate the speed profile
    top = boxes[:, 5:8] / 1000
    time = boxes[:, 0]
    velocity_profile = horizontal_velocity(top, time)

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches((10, 3))
    im_h = ax[0].imshow(frames[0, ...])
    ax[1].plot(time, velocity_profile)
    cursor = ax[1].axvline(time[0], color='k')
    plt.tight_layout()
    plt.pause(0.001)

    if export_video:
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15)
        with writer.saving(fig, 'export_{}.mp4'.format(id), 100):
            for i in range(frames.shape[0]):
                im_h.set_data(frames[i, ...])
                cursor.set_xdata(time[i])
                plt.pause(0.001)
                writer.grab_frame()
    else:
        for i in range(frames.shape[0]):
            im_h.set_data(frames[i, ...])
            cursor.set_xdata(time[i])
            plt.pause(0.001)


def play_sts_sequence(id, obj, export_video=False):
    frames = load_frames(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)
    boxes = load_boxes(id, obj.meta_list, obj.data_mode, obj.data_path, obj.zips_list, obj.in_which_zip)

    # Pre-calculate speed of ascent
    cy = boxes[:, 6] / 1000  # Use the y coordinate of the upper edge of the 3D bounding box
    time = boxes[:, 0]
    cy_smooth = savgol_filter(cy, 11, 3, mode='interp')
    deriv = savgol_deriv(cy, time, 11, 3)
    soa, _ = speed_of_ascent(cy, time, obj.target)

    f = np.argmin(np.abs(deriv - soa))
    frames[f-3:f+3, :, :, 1:3] = 0

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches((10, 3))
    im_h = ax[0].imshow(frames[0, ...])
    ax[1].plot(time, cy_smooth)
    cursor = ax[1].axvline(time[0], color='k')
    t = ax[1].twinx()
    t.plot(time, deriv, color='orange')
    t.axhline(soa, color='red')
    plt.tight_layout()
    plt.pause(0.001)

    if export_video:
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15)
        with writer.saving(fig, 'export_{}.mp4'.format(id), 100):
            for i in range(frames.shape[0]):
                im_h.set_data(frames[i, ...])
                cursor.set_xdata(time[i])
                plt.pause(0.001)
                writer.grab_frame()
    else:
        for i in range(frames.shape[0]):
            im_h.set_data(frames[i, ...])
            cursor.set_xdata(time[i])
            plt.pause(0.001)


def play_bson_silhouette(days_folder, start, end, room):
    'G:\HouseData-Sphere\4954\backups'
    day = '{}-{:02}-{:02}'.format(start.year, start.month, start.day)
    bson_path = os.path.join(days_folder, day, 'sphere', 'VID.bson')
    if os.path.exists(bson_path):
        use_bzip = False
    else:
        use_bzip = True
        bson_path += '.bz2'

    bson_data = BsonFile(bson_path)
    room_uid = read_room_uid(os.path.dirname(bson_path), room, use_bzip)
    json_data = []
    for entry in tqdm(bson_data.data):
        entry_time = entry['bt']
        entry_room = entry['uid']
        if entry_room == room_uid and entry_time >= start and entry_time <= end:
            json_data.append(entry)

    silhouettes = [bf['e'][0]['v'] for bf in json_data if bf['e'][0]['n'] == 'silhouette']
    sample = b64_image_to_numpy(silhouettes[0])
    video = np.zeros((len(silhouettes), sample.shape[0], sample.shape[1]), dtype=np.uint8)
    for i, silhouette in enumerate(tqdm(silhouettes)):
        video[i, ...] = b64_image_to_numpy(silhouette)
    video_name = 'bson_{}.mp4'.format(start).replace(':', '.')
    vwrite(video_name, video)
    print('Frames from {} to {} were saved to {}'.format(start, end, video_name))


def play_bson_silhouette_and_wearable(days_folder, start, end, room, export_video=False):
    day = '{}-{:02}-{:02}'.format(start.year, start.month, start.day)
    vid_path = os.path.join(days_folder, day, 'sphere', 'VID.bson')
    wear_path = os.path.join(days_folder, day, 'sphere', 'WEAR.bson')
    if os.path.exists(vid_path):
        use_bzip = False
    else:
        use_bzip = True
        vid_path += '.bz2'
        wear_path += '.bz2'

    vid_data = BsonFile(vid_path)
    wear_data = BsonFile(wear_path)
    room_uid = read_room_uid(os.path.dirname(vid_path), room, use_bzip)
    vid_json = []
    for entry in tqdm(vid_data.data):
        entry_time = entry['bt']
        entry_type = entry['e'][0]['n']
        entry_room = entry['uid']
        if entry_type == 'silhouette' and entry_room == room_uid and entry_time >= start and entry_time <= end:
            vid_json.append(entry)

    wear_json = []
    for entry in tqdm(wear_data.data):
        entry_time = entry['bt']
        entry_participant = entry['uid']
        if entry_time >= start and entry_time <= end:
            wear_json.append(entry)

    gw = 0
    wear_time = [bf['bt'] for bf in wear_json if 'e' in bf.keys()]
    wear_accel = [bf['e'][gw]['v'] for bf in wear_json if 'e' in bf.keys()]
    ord = np.argsort(wear_time)
    wear_time = np.array([wear_time[bf] for bf in ord])
    wear_accel = np.array([wear_accel[bf] for bf in ord])

    vid_silhou = [bf['e'][0]['v'] for bf in vid_json if bf['e'][0]['n'] == 'silhouette']
    vid_time = [bf['bt'] for bf in vid_json if bf['e'][0]['n'] == 'silhouette']
    sample = b64_image_to_numpy(vid_silhou[0])

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    im_h = ax[0].imshow(sample, cmap='gray')
    ax[1].plot(wear_time, wear_accel[:, 0])
    ax[1].plot(wear_time, wear_accel[:, 1])
    ax[1].plot(wear_time, wear_accel[:, 2])
    cu_h = ax[1].axvline(wear_time[0], color='k')

    if export_video:
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15)
        with writer.saving(fig, 'wear_silu.mp4', 100):
            for i, silhouette in enumerate(tqdm(vid_silhou)):
                im_h.set_data(b64_image_to_numpy(silhouette))
                cu_h.set_xdata(vid_time[i])
                plt.pause(0.01)
                writer.grab_frame()
    else:
        for i, silhouette in enumerate(tqdm(vid_silhou)):
            im_h.set_data(b64_image_to_numpy(silhouette))
            cu_h.set_xdata(vid_time[i])
            plt.pause(0.01)




def day_bson_average(days_folder, day, room, fast=False):
    """Extract all the silhouettes from a room on a single day and shows the average of the edges in a log image"""
    bson_path = os.path.join(days_folder, day, 'sphere', 'VID.bson')
    if os.path.exists(bson_path):
        use_bzip = False
    else:
        use_bzip = True
        bson_path += '.bz2'

    bson_data = BsonFile(bson_path)
    room_uid = read_room_uid(os.path.dirname(bson_path), room, use_bzip)
    json_data = []
    for entry in tqdm(bson_data.data):
        entry_type = entry['e'][0]['n']
        entry_room = entry['uid']
        if entry_type == 'silhouette' and entry_room == room_uid:
            json_data.append(entry)

    silhouettes = [bf['e'][0]['v'] for bf in json_data if bf['e'][0]['n'] == 'silhouette']
    if fast:
        less = np.linspace(0, len(silhouettes)-1, 1000, dtype=np.int)
        silhouettes = [silhouettes[bf] for bf in less]
    if len(silhouettes) > 0:
        sample = b64_image_to_numpy(silhouettes[0])
        average = np.zeros((sample.shape[0], sample.shape[1]))
        for i, silhouette in enumerate(tqdm(silhouettes)):
            img = b64_image_to_numpy(silhouette)/255
            gy = np.vstack((np.zeros((1, sample.shape[1])), np.diff(img, axis=0)))
            gx = np.hstack((np.zeros((sample.shape[0], 1)), np.diff(img, axis=1)))
            img = np.hypot(gx, gy)
            average += img

        plt.figure()
        plt.imshow(np.log(average), cmap='terrain')
        plt.colorbar()
        plt.title(bson_path)
        return average
    else:
        print('No silhouettes found')
        return None