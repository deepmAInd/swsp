"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""
from typing import Dict, Callable, Any
import pandas as pd
from .base import (
    butter_lowpass_filter,
    fs_dict,
    filterSignalFIR,
    eda_stats,
    SubjectData,
    get_net_accel,
    get_window_stats,
    get_peak_freq,
    get_slope,
)
import gc
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


WINDOW_IN_SECONDS = 30
WINDOW_ADDITION = 15
WINDOW_INTERVAL = 15
WINDOW_LENGTH = WINDOW_IN_SECONDS + WINDOW_ADDITION

label_dict = {"baseline": 1, "stress": 2, "amusement": 0}
int_to_label = {1: "baseline", 2: "stress", 0: "amusement"}
feat_names = None


def _compute_features(e4_data_dict, labels, norm_type=None):

    # Dataframes for each sensor type
    eda_df = pd.DataFrame(e4_data_dict["EDA"], columns=["EDA"])
    bvp_df = pd.DataFrame(e4_data_dict["BVP"], columns=["BVP"])
    acc_df = pd.DataFrame(e4_data_dict["ACC"], columns=["ACC_x", "ACC_y", "ACC_z"])
    temp_df = pd.DataFrame(e4_data_dict["TEMP"], columns=["TEMP"])
    label_df = pd.DataFrame(labels, columns=["label"])
    resp_df = pd.DataFrame(e4_data_dict["Resp"], columns=["Resp"])

    # Filter EDA
    eda_df["EDA"] = butter_lowpass_filter(eda_df["EDA"], 1.0, fs_dict["EDA"], 6)

    # Filter ACM
    for _ in acc_df.columns:
        acc_df[_] = filterSignalFIR(acc_df.values)

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / fs_dict["EDA"]) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / fs_dict["BVP"]) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / fs_dict["ACC"]) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / fs_dict["TEMP"]) * i for i in range(len(temp_df))]
    label_df.index = [(1 / fs_dict["label"]) * i for i in range(len(label_df))]
    resp_df.index = [(1 / fs_dict["Resp"]) * i for i in range(len(resp_df))]
    # print(eda_df)

    # Change indices to datetime
    eda_df.index = pd.to_datetime(eda_df.index, unit="s")
    bvp_df.index = pd.to_datetime(bvp_df.index, unit="s")
    temp_df.index = pd.to_datetime(temp_df.index, unit="s")
    acc_df.index = pd.to_datetime(acc_df.index, unit="s")
    label_df.index = pd.to_datetime(label_df.index, unit="s")
    resp_df.index = pd.to_datetime(resp_df.index, unit="s")

    # New EDA features
    r, p, t, l, d, e, obj = eda_stats(eda_df["EDA"])
    eda_df["EDA_phasic"] = r
    eda_df["EDA_smna"] = p
    eda_df["EDA_tonic"] = t

    # Combined dataframe
    df = eda_df.join(bvp_df, how="outer")
    df = df.join(temp_df, how="outer")
    df = df.join(acc_df, how="outer")
    df = df.join(label_df, how="outer")

    df["label"] = df["label"].fillna(method="bfill")

    df.reset_index(drop=True, inplace=True)

    df_ = df.drop(columns=["label"])
    df_ = df_.dropna(how="all")

    df_merged = df_.join(df["label"], how="left")

    if norm_type == "std":
        # std norm
        df_merged = (df_merged - df_merged.mean()) / df_merged.std()
    elif norm_type == "minmax":
        # minmax norm
        df_merged(df_merged - df_merged.min()) / (df_merged.max() - df_merged.min())

    # Group by
    grouped = df_merged.groupby("label")

    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)
    return grouped, baseline, stress, amusement


def _get_samples(data, label) -> pd.DataFrame:
    global feat_names
    global WINDOW_LENGTH
    global WINDOW_INTERVAL

    samples = []
    # Using label freq (64 Hz) as our reference frequency due to it being the largest
    # and thus encompassing the lesser ones in its resolution.
    window_len = fs_dict["BVP"] * WINDOW_LENGTH
    window_int = fs_dict["BVP"] * WINDOW_INTERVAL
    i = 0
    data.head()
    while (window_int * i + window_len) < len(data):

        # Get window of data
        w = data[window_int * i : window_int * i + window_len]

        # Add/Calc rms acc
        # w['net_acc'] = get_net_accel(w)
        w = pd.concat([w, get_net_accel(w)], names=["acc_net"])
        # w.columns = ['net_acc', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP',
        #           'EDA', 'EDA_phasic', 'EDA_smna', 'EDA_tonic', 'TEMP',
        #         'label']
        # print(w.head())

        cols = list(w.columns)
        cols[0] = "net_acc"
        w.columns = cols

        # Calculate stats for window
        wstats = get_window_stats(data=w, label=label)

        # Seperating sample and label
        x = pd.DataFrame(wstats).drop("label", axis=0)
        y = x["label"][0]
        x.drop("label", axis=1, inplace=True)

        if feat_names is None:
            feat_names = []
            for row in x.index:
                for col in x.columns:
                    feat_names.append("_".join([str(row), str(col)]))

        # sample df
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names
        wdf = pd.concat([wdf, pd.DataFrame({"label": y}, index=[0])], axis=1)

        # More feats
        wdf["BVP_peak_freq"] = get_peak_freq(w["BVP"].dropna())
        wdf["TEMP_slope"] = get_slope(w["TEMP"].dropna())

        samples.append(wdf)
        i += 1

    return pd.concat(samples)


def make_patient_data(
    patient_loader: Dict[str, Callable[..., Any]], int_out_path: str
) -> None:
    global WINDOW_IN_SECONDS
    for subject, loader_func in patient_loader.items():
        # Make subject data object for Sx
        subject = SubjectData(data=loader_func(), name=subject)

        # Empatica E4 data - now with resp
        e4_data_dict = subject.get_wrist_data()

        # norm type
        norm_type = None

        # The 3 classes we are classifying
        grouped, baseline, stress, amusement = _compute_features(
            e4_data_dict, subject.labels, norm_type
        )

        baseline_samples = _get_samples(baseline, 1)
        stress_samples = _get_samples(stress, 2)
        amusement_samples = _get_samples(amusement, 0)

        all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
        all_samples = pd.concat(
            [all_samples.drop("label", axis=1), pd.get_dummies(all_samples["label"])],
            axis=1,
        )
        all_samples.to_csv(f"{int_out_path}/{subject.name}_feats_4.csv")
        gc.collect()


def combine_patients(patients_loader) -> pd.DataFrame:
    df_list = []
    for subject, loader_func in patients_loader.items():
        df = loader_func()
        df["subject"] = subject
        df_list.append(df)

    df = pd.concat(df_list)

    df["label"] = (
        df["0"].astype(str) + df["1"].astype(str) + df["2"].astype(str)
    ).apply(lambda x: x.index("1"))
    df.drop(["0", "1", "2"], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    counts = df["label"].value_counts()
    print("Number of samples per class:")
    for label, number in zip(counts.index, counts.values):
        print(f"{int_to_label[label]}: {number}")
    return df
