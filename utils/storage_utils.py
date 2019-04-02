import os
import zipfile

from sphere.utils.image_utils import img_to_buffer

class ZipStore:
    def __init__(self, cfg, output_folder):
        self.cfg = cfg
        self.output_folder = output_folder
        self.name_i = 0
        self.seq_i = 0
        self.zip_video = None
        self.zip_meta = None
        self.create_next_zip()

    # noinspection PyTypeChecker
    def create_next_zip(self):
        zip_name_video = os.path.join(self.output_folder, (self.cfg['zip_template'] % self.name_i) + '_video.zip')
        zip_name_meta = os.path.join(self.output_folder, (self.cfg['zip_template'] % self.name_i) + '_meta.zip')
        self.zip_video = zipfile.ZipFile(zip_name_video, mode='w', compression=self.cfg['zip_comp'])
        self.zip_meta = zipfile.ZipFile(zip_name_meta, mode='w', compression=self.cfg['zip_comp'])
        self.seq_i = 0

    def add_entry(self, video, video_name, meta, meta_name, day):
        # Add frames to the archive
        for fi in range(video.shape[0]):
            # Save each frame as a jpeg into the buffer
            buffer = img_to_buffer(video[fi, :, :])
            file_path = os.path.join('Videos', day, video_name, '%04d.jpg' % fi)
            self.zip_video.writestr(file_path, buffer)

        self.zip_meta.writestr(meta_name, meta)
        self.seq_i += 1
        self._check_full()

    def _check_full(self):
        if self.seq_i == self.cfg['zip_n_max']:
            print('Zip {} completed'.format(self.name_i))
            self.zip_video.close()
            self.zip_meta.close()
            self.name_i += 1

            self.create_next_zip()

    def close(self):
        self.zip_video.close()
        self.zip_meta.close()

