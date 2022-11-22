import os
import pickle


class SubjectData:
    def __init__(self, data, name):
        self.name = name
        self.subject_keys = ["signal", "label", "subject"]
        self.signal_keys = ["chest", "wrist"]
        self.chest_keys = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]
        self.wrist_keys = ["ACC", "BVP", "EDA", "TEMP"]
        self.data = data
        self.labels = self.data["label"]

    def get_wrist_data(self):
        data = self.data["signal"]["wrist"]
        data.update({"Resp": self.data["signal"]["chest"]["Resp"]})
        return data

    def get_chest_data(self):
        return self.data["signal"]["chest"]

    def extract_features(self):  # only wrist
        results = {
            key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
            for key in self.wrist_keys
        }
        return results
