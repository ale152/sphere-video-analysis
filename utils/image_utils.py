import base64
import numpy as np

from io import BytesIO
from PIL import Image
from scipy.misc import imresize


def b64_image_to_numpy(b64_string):
    # Convert the b64 string into image
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(b64_string))
    pimg = Image.open(sbuf)
    img = np.array(pimg)
    return img


def img_to_buffer(img, format='jpeg'):
        buffer = BytesIO()
        image = Image.fromarray(img, 'L')
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()


def crop_resize_image(image, box_2d, target_size):
    cropbox = image[box_2d[1]:box_2d[3], box_2d[0]:box_2d[2]]
    ar = cropbox.shape[0] / cropbox.shape[1]
    # Vertically elongated
    if ar > 1:
        # The height will match the target size
        new_size = [target_size, np.round(target_size / ar)]
    # Horizontally elongated
    else:
        # The width will match the target size
        new_size = [np.round(target_size * ar), target_size]

    new_size = np.round(new_size).astype('int')
    # If the image is smaller than the target size, don't resize it
    if np.all([bf < target_size for bf in cropbox.shape]):
        new_size = cropbox.shape
    else:
        cropbox = imresize(cropbox, new_size, interp='nearest')

    imgbox = np.zeros((target_size, target_size))
    first_row = np.round(target_size / 2 - new_size[1] / 2).astype('int')
    first_col = np.round(target_size / 2 - new_size[0] / 2).astype('int')
    imgbox[first_col:first_col + new_size[0], first_row:first_row + new_size[1]] = cropbox
    return imgbox
