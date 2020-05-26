import numpy as np
from scipy.signal import butter, lfilter


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


x = np.arange(10)
print(x)
print(x[-7:])
