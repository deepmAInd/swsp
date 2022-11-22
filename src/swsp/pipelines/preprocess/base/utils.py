import scipy.signal as scisig
import scipy.stats
import numpy as np

from .cvx_optx import cvxEDA

fs_dict = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "label": 700, "Resp": 700}


def eda_stats(y):
    Fs = fs_dict["EDA"]
    yn = (y - y.mean()) / y.std()
    [r, p, t, l, d, e, obj] = cvxEDA(yn, 1.0 / Fs)
    return [r, p, t, l, d, e, obj]


def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y


def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series)
    slope = linreg[0]
    return slope


def get_window_stats(data, label=-1):
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {
        "mean": mean_features,
        "std": std_features,
        "min": min_features,
        "max": max_features,
        "label": label,
    }
    return features


def get_net_accel(data):
    return (data["ACC_x"] ** 2 + data["ACC_y"] ** 2 + data["ACC_z"] ** 2).apply(
        lambda x: np.sqrt(x)
    )


def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq


# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/AccelerometerFeatureExtractionScript.py
def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff / (fs_dict["ACC"] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)
