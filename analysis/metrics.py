import numpy as np
from warnings import warn

from sphere.utils import find_peaks, signal_to_noise, savgol_deriv


def speed_of_ascent(y_box, time, target, smooth_kernel=11, smooth_poly=3):
    """Calculate the speed of ascent given the vertical position of the 3D bounding box"""
    # First, filter the data and calculate the derivatives
    # cy_smooth = savgol_filter(y_box, smooth_kernel, smooth_poly, mode='interp')
    deriv = savgol_deriv(y_box, time, smooth_kernel, smooth_poly)

    if target == 'stand up':
        deriv = + deriv
    elif target == 'sit down':
        deriv = - deriv
    else:
        raise Exception('The only targets supported are "stand up" and "sit down"')

    # Find the peaks
    pks, pks_val = find_peaks(deriv)
    snr = signal_to_noise(pks_val)

    if len(pks_val) > 0:
        speed = pks_val.max()
    else:
        warn('The signal provided does not have any peak. Returning speed of ascent NaN')
        speed = np.nan

    return speed, snr


def walking_speed(horizontal_speed, speed_thr, duration_thr):
    """lorem ipsum"""
    walking = (horizontal_speed > speed_thr).astype('float')
    bounded = np.insert(walking, 0, 0)
    bounded = np.append(bounded, 0)
    right = np.where(np.diff(bounded) < 0)[0]
    left = np.where(np.diff(bounded) > 0)[0]
    duration = right - left
    if len(duration) > 0 and duration.max()/100 > duration_thr:
        pos = np.argmax(duration)
        walking_speed = np.mean(horizontal_speed[left[pos]:right[pos]])
        return walking_speed, duration.max(), left[pos]
    else:
        return np.nan, np.nan, 0


def horizontal_velocity(top, time, smooth_kernel=11, smooth_poly=3):
    """Calculate the horizontal velocity using the x and z coordinates of the top vertex of the bounding box"""
    speed_x = savgol_deriv(top[:, 0], time, smooth_kernel, smooth_poly)
    speed_z = savgol_deriv(top[:, 2], time, smooth_kernel, smooth_poly)
    speed = np.sqrt(np.square(speed_x) + np.square(speed_z))
    return speed