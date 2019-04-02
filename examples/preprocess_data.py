import os
import zipfile
import numpy as np
from tqdm import tqdm

from utils.image_utils import b64_image_to_numpy, crop_resize_image, img_to_buffer
from utils.storage_utils import ZipStore
from data_loader.mongo_loader import HouseData, VidSequence


if __name__ == '__main__':
    list_houses_path = r'J:\HouseData-Sphere'
    # data_path = r'G:\HouseData-Sphere\4954\backups'  # The folder with the days folders
    output = r'G:\STS_zipped_overlap_50_new'  # Where video sequences will be saved

    cfg = {'dt': 0.3,  # Threshold (seconds) for considering frames consecutive
           'room': 'hal',  # Either 'kit', 'hal', or 'liv'
           'featid_only': False,  # Only saves the frames containing ReID feature
           'text_verbose': False,
           'zip_n_max': 1000,  # Number of files per zip. 1000 results in 100-250 MB zip files
           'zip_template': 'chunk_%d',
           'zip_comp': zipfile.ZIP_STORED,
           'vid_ovlap': 2,  # Number of overlapping videos: (1-1/N)*100 (% overlap)
           'vid_size': 100,  # Target size for the reshaped boxes
           'vid_frames': 100  # Maximum length of the sequence (frames)
           }

    list_houses = ['3099']

    for house in list_houses:
        print('Processing {} room {}'.format(house, cfg['room']))
        print('Processing house {}'.format(house))
        data_path = os.path.join(list_houses_path, house, 'backups')
        output_folder = os.path.join(output, house, cfg['room'])

        house_data = HouseData(data_path, output_folder, cfg, mode='VID')

        # Initialise the first zip file
        zip_store = ZipStore(cfg, output_folder)

        # Loop over days
        tq = tqdm(house_data.days_list, smoothing=0.1)
        for day in tq:
            house_data.open_day(day)
            if house_data.data is None:
                continue

            # The last_silu_base_time is updated every time a silhouette is found. It is then used to match silhouettes
            # and bounding boxes
            last_sil_time = 0
            img = None

            # Initialise the sequence. This is a list of dictionaries, each one containing all the information from the
            # database (silhouette, bounding box, base time, ...)
            tracked = {'user_id': [], 'sequence': []}

            ############################################################################################################
            # Loop over the data (silhouettes/bounding boxes)
            ############################################################################################################
            for vid_entry in house_data.data:
                # Give variables a sensible name
                entry_time = vid_entry['bt']
                entry_type = vid_entry['e'][0]['n']
                entry_room = vid_entry['uid']

                # Check if the entry is a silhouette from the target room
                if entry_type == 'silhouette' and entry_room == house_data.room_uid:
                    last_sil_time = entry_time
                    # Convert the b64 string into image
                    img = b64_image_to_numpy(vid_entry['e'][0]['v'])

                # Check if the entry is a bounding box
                elif entry_time == last_sil_time and entry_room == house_data.room_uid:
                    bbox_2d = np.array(vid_entry['e'][2]['v'])
                    bbox_3d = np.array(vid_entry['e'][4]['v'])
                    entry_reid = vid_entry['e'][8]['v']
                    entry_userid = vid_entry['e'][1]['v']

                    # Check if the bbox is valid. Some boxes have negative xy
                    if np.any(bbox_2d < 0):
                        continue

                    # Only process frames with feature reid, if requested
                    if cfg['featid_only'] and entry_reid == []:
                        continue

                    # Crop image from bbox and resize it
                    img_crop = crop_resize_image(img, bbox_2d, cfg['vid_size'])

                    # If the user is not in the lists
                    try:
                        found_id = tracked['user_id'].index(entry_userid)
                    except ValueError:
                        found_id = None

                    if found_id is None:
                        # Add the user to the list
                        new_sequence = VidSequence(entry_userid, cfg, img_crop, entry_time, bbox_2d, bbox_3d)
                        tracked['user_id'].append(entry_userid)
                        tracked['sequence'].append(new_sequence)
                    else:
                        tracked['sequence'][found_id].update(img_crop, entry_time, bbox_2d, bbox_3d)
                        tracked['sequence'][found_id].check_complete(zip_store, house_data.room_uid, day)
                        # If a sequence is lost, it'll stay in memory until the "end of the day"

        # Close the last zip files
        zip_store.close()
