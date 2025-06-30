import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler


class ScalerManager:
    def __init__(self, scaler_config_path):
        with open(scaler_config_path, "r") as f:
            self.config = json.load(f)
        self.scalers = self._build_scalers()

    def _build_scalers(self):
        scalers = {}
        for key, bounds in self.config.items():
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(np.array(bounds).reshape(-1, 1))
            scalers[key] = scaler
        return scalers

    def scale(self, key, value):
        return self.scalers[key].transform(np.array([[value]]))[0, 0]

    def unscale(self, key, value):
        return self.scalers[key].inverse_transform(np.array([[value]]))[0, 0]
