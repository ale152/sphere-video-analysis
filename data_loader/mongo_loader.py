import os
import bson
import bz2
import numpy as np
from io import BytesIO
from datetime import datetime

from sphere.utils.mongo_utils import open_bzip, read_room_uid

class HouseData:
    def __init__(self, day_folders_path, output_folder, cfg, mode):
        """Define an object to load data from the bson mongo dumps. Data is accessed by creating the object,
        opening a day and using the self.data"""
        self.day_folders_path = day_folders_path
        self.cfg = cfg
        self.mode = mode

        self.room_uid = None
        self.data = None

        # List days and convert them into datetime
        self.days_list = os.listdir(self.day_folders_path)
        self.days_list = [bf for bf in self.days_list if bf.startswith('20')]
        self.date_list = [datetime.strptime(bf, '%Y-%m-%d') for bf in self.days_list]

        # Create output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def open_day(self, day):
        if self.mode == 'VID':
            bson_file = 'VID.bson'
        elif self.mode == 'WEAR':
            bson_file = 'WEAR.bson'
        else:
            raise NotImplemented

        bson_path = os.path.join(self.day_folders_path, day, 'sphere', bson_file)
        if os.path.exists(bson_path):
            use_bzip = False
        else:
            bson_path += '.bz2'
            if os.path.exists(bson_path):
                use_bzip = True
            else:
                print('File {} not found. Skipping this folder...'.format(bson_path))
                self.data = None
                return

        # Check the file size to make sure there is data
        fsize = os.stat(bson_path).st_size
        if fsize < 1000:
            print('Skipping file %s because is empty (less than 1 kB)' % day)
            self.data = None
            return
        else:
            print('Loading file %s of size %.1f MB' % (day, fsize/1e6))

        # The room data is only needed in video mode, not wearable
        if self.mode == 'VID':
            room_path = os.path.join(self.day_folders_path, day, 'sphere')
            bf = read_room_uid(room_path, self.cfg['room'], use_bzip)
            if bf is not None:
                self.room_uid = bf

        # Open the data file
        fid = open_bzip(bson_path, use_bzip)

        # Dump the file into memory
        fid.seek(0, os.SEEK_END)
        size = fid.tell()
        fid.seek(0)
        # Allocate the buffer in memory
        f_mem = BytesIO(b'\x00' * size)
        f_mem.seek(0)
        f_mem.write(fid.read())
        f_mem.seek(0)
        fid.close()

        self.data = bson.decode_file_iter(f_mem)  # TODO Possible data leak if I don't close f_mem?


class BsonFile:
    def __init__(self, bson_path):
        if os.path.splitext(bson_path)[1] == '.bz2':
            self.use_bzip = True
        else:
            self.use_bzip = False

        fid = self._open_bzip(bson_path)

        # Dump the file into memory
        fid.seek(0, os.SEEK_END)
        size = fid.tell()
        fid.seek(0)
        # Allocate the buffer in memory
        f_mem = BytesIO(b'\x00' * size)
        f_mem.seek(0)
        f_mem.write(fid.read())
        f_mem.seek(0)
        fid.close()

        self.data = bson.decode_file_iter(f_mem)  # TODO Possible data leak if I don't close f_mem?

    def _open_bzip(self, file):
        if self.use_bzip:
            fid = bz2.open(file)
        else:
            fid = open(file, 'rb')
        return fid


class VidSequence:
    """A Sequence object contain a contiguous sequence of frames and bounding boxes extraced from the Sphere database"""
    def __init__(self, user_id, cfg, image, time, box_2d, box_3d):
        ov_n, fr_n = cfg['vid_ovlap'], cfg['vid_frames']
        self.cfg = cfg
        self.user_id = user_id
        self.image = np.zeros((ov_n, fr_n, cfg['vid_size'], cfg['vid_size']), dtype=np.uint8)
        self.time = np.zeros((ov_n, fr_n), dtype=object)
        self.box_2d = np.zeros((ov_n, fr_n, 4))
        self.box_3d = np.zeros((ov_n, fr_n, 6))
        self.next_frames = np.zeros((cfg['vid_ovlap']), dtype=np.int)
        self.frame_index = 0

        # Update the first overlapping video
        self._set(0, 0, image, time, box_2d, box_3d)
        self.next_frames[0] = 1

    def update(self, image, time, box_2d, box_3d):
        """Update the sequence with the next data"""
        # First, check that the entry frame is not too far from the last frame recorded, otherwise the sequence
        # is considered lost. This check is done on the first overlapping video, since it is the most recent one
        time_last_frame = self.time[0, self.next_frames[0] - 1]
        if (time - time_last_frame).total_seconds() > self.cfg['dt']:
            # The sequence is lost. Start a new sequence and record the entry frame in the first overlapping video
            self._set(0, 0, image, time, box_2d, box_3d)
            # The overlapping videos are just reset by changing the frame index to zero
            self.next_frames[0] = 1
            self.next_frames[1:] = 0
            # The master frame index is set to zero to restore the overlap delay functionality
            self.frame_index = 0
        else:
            # The frame is sequential, increase the master frame index
            self.frame_index += 1

            # Loop over the overlapping videos
            ov_n, fr_n = self.cfg['vid_ovlap'], self.cfg['vid_frames']
            for ov_i in range(ov_n):
                # Calculate the number of frames of delay from the master index before this overlapping clip should
                # start
                delay = fr_n / ov_n * ov_i
                if self.frame_index >= delay:
                    # Update the sequence
                    fr_i = self.next_frames[ov_i]
                    self._set(ov_i, fr_i, image, time, box_2d, box_3d)
                    self.next_frames[ov_i] += 1

    def check_complete(self, zip_store, room_uid, day):
        """Check if the sequence is complete and store it to a zip file"""
        ov_n, fr_n = self.cfg['vid_ovlap'], self.cfg['vid_frames']
        for ov_i in range(ov_n):
            if self.next_frames[ov_i] == fr_n:
                # Prepare the video data to be stored
                video_name = '%s_%d_%s' % (
                    room_uid,
                    self.user_id,
                    self.time[ov_i, -1].isoformat().replace(':', '.'))
                video = self.image[ov_i, ...]

                # Prepare the metadata to be stored
                meta = BytesIO()
                string_time = np.array([str(bf) for bf in self.time[ov_i, :]])[..., None]
                metadata = np.hstack((string_time, self.box_2d[ov_i, ...], self.box_3d[ov_i, ...]))
                # noinspection PyTypeChecker
                np.savetxt(meta, metadata,
                           fmt='%s', delimiter=',', newline='\n',
                           header='Time %Y-%m-%d %H:%M:%S.%f, 2dBB [4], 3dBB [6]')
                meta = meta.getvalue()

                csv_name = video_name + '.csv'
                meta_name = os.path.join('Videos', day, video_name, csv_name)

                zip_store.add_entry(video, video_name, meta, meta_name, day)
                self.next_frames[ov_i] = 0

    def _set(self, ov_i, fr_i, image, time, box_2d, box_3d):
        """Set the data for a specific frame of a specific overlapping clip"""
        self.image[ov_i, fr_i, :, :] = image
        self.time[ov_i, fr_i] = time
        self.box_2d[ov_i, fr_i, :] = box_2d
        self.box_3d[ov_i, fr_i, :] = box_3d
