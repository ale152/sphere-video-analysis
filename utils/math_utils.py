from warnings import warn
from scipy.signal import savgol_filter, medfilt
import numpy as np


def find_peaks(y):
    """Find peak locations in the vector y. Peaks are defined as points preceded and followed by a lower value"""
    diff_y = np.diff(y)
    sign_diff = np.sign(diff_y)
    diff_sign_diff = np.diff(sign_diff)
    pks = np.where(diff_sign_diff == -2)[0] + 1
    return pks, y[pks]


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


def savgol_deriv(y, t, kernel, poly):
    y = np.concatenate((y[:5][::-1], y, y[-5:][::-1]))
    y = medfilt(y, 5)
    y = y[5:-5]
    diff_y = savgol_filter(y, kernel, poly, deriv=1, mode='interp')
    diff_t = np.diff(t)
    diff_t = np.append(diff_t, diff_t[-1])
    # diff_t = savgol_filter(t, kernel, poly, deriv=1, mode='interp')
    deriv = diff_y / diff_t
    return deriv
