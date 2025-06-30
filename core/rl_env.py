import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import pandas as pd
from typing import Callable
from .utils.scaler import ScalerManager
from .models.model_4r2c import ThermalModel4R2C


class BuildingEnv(gym.Env):
    def __init__(
        self,
        config_path: str,
        scaler_config_path: str,
        reward_fn: Callable[[dict, float, dict], float],
        test_mode: bool = False,
    ):
        super().__init__()
        self.test_mode = test_mode

        with open(config_path, "r") as f:
            config = json.load(f)

        self.state_keys = config["state_keys"]
        self.action_type = config["action_type"]
        self.action_space_config = config["action_space"]

        self.reward_fn = reward_fn
        self.scaler_manager = ScalerManager(scaler_config_path)

        self.df = None
        self.thermal_model = None
        self.current_day_idx = None
        self.current_episode_df = None
        self.time_step = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.state_keys),), dtype=np.float32
        )

        if self.action_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        elif self.action_type == "discrete":
            self.action_space = spaces.Discrete(len(self.action_space_config["values"]))
        else:
            raise ValueError("Invalid action_type")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0

        if self.df is None or self.thermal_model is None:
            raise RuntimeError(
                "Dataframe and thermal model must be set using set_inputs() before calling reset()."
            )

        self.thermal_model.reset()
        self.prev_load = 0.0
        self.ep_energy_consumed = 0.0
        self.ep_co2_emission = 0.0
        self.ep_energy_cost = 0.0
        self.ep_thermal_discomfort = 0.0

        if self.test_mode:
            # Test through the whole scenario
            start_idx = 0
            end_idx = len(self.df)
        else:
            # Pick a random day
            total_days = len(self.df) // 96
            self.current_day_idx = np.random.randint(0, total_days)
            start_idx = self.current_day_idx * 96
            end_idx = start_idx + 96
        self.current_episode_df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)

        self.prev_load = 0.0
        return self._get_obs(), {}

    def step(self, action):
        prev_state_dict = self.unscaled_state_dict
        if self.action_type == "discrete":
            action = np.array([self.action_space_config["values"][action]])
            act_min, act_max = (
                self.action_space_config["values"][0],
                self.action_space_config["values"][-1],
            )
        else:
            act_min, act_max = self.action_space_config["range"]

        act = (action[0] + 1) / 2 * (act_max - act_min) + act_min

        tamb, qirr, pelec, price, co2 = self._fetch_timestep_data(self.time_step)
        ac_cooling_power = self.thermal_model.step(act, tamb, qirr, pelec)

        # Store the last observed load and update metrics
        self.prev_load = pelec + ac_cooling_power
        self.update_metrics(co2=co2, price=price)

        # Fetch the next state (except for the fixed load, which is unknown)
        self.time_step += 1

        if self.test_mode:
            done = self.time_step >= len(self.df)
        else:
            done = self.time_step >= 96
        if done:
            info = {
                "ep_energy_consumed": self.ep_energy_consumed,
                "ep_co2_emission": self.ep_co2_emission,
                "ep_energy_cost": self.ep_energy_cost,
                "ep_thermal_discomfort": self.ep_thermal_discomfort,
            }
            tamb, qirr, pelec, price, co2 = self._fetch_timestep_data(
                self.time_step - 1
            )
        else:
            info = {}
            tamb, qirr, pelec, price, co2 = self._fetch_timestep_data(self.time_step)

        state_dict = {
            "time": (self.time_step % 96) / 96.0,
            "t_in": self.thermal_model.t_in,
            "t_amb": tamb,
            "q_irr": qirr,
            "prev_load": self.prev_load,
            "curr_price": price,
            "curr_co2": co2,
        }
        state = self._get_obs(state_dict)
        reward = self.reward_fn(prev_state_dict, action[0], self.unscaled_state_dict)

        return state, reward, done, False, info

    def _get_obs(self, state_dict=None):
        if state_dict is None:
            row = self.current_episode_df.iloc[self.time_step]
            state_dict = {
                "time": (self.time_step % 96) / 96.0,
                "t_in": self.thermal_model.t_in,
                "t_amb": row["Tamb_C"],
                "q_irr": row["Qirr_W_m2"],
                "prev_load": self.prev_load,
                "curr_price": row["USEP_SGD_MWh"],
                "curr_co2": row["CO2_gCO2eq_kWh"],
            }
        self.unscaled_state_dict = {k: state_dict[k] for k in self.state_keys}
        obs = [
            self.scaler_manager.scale(key, state_dict[key]) for key in self.state_keys
        ]
        return np.array(obs, dtype=np.float32)

    def _fetch_timestep_data(self, time_step: int):
        row = self.current_episode_df.iloc[time_step]
        tamb = row["Tamb_C"]
        qirr = row["Qirr_W_m2"]
        pelec = row["Pelec_W"]
        price = row["USEP_SGD_MWh"]
        co2 = row["CO2_gCO2eq_kWh"]
        return tamb, qirr, pelec, price, co2

    def set_inputs(self, df: pd.DataFrame, thermal_model: ThermalModel4R2C):
        self.df = df
        self.thermal_model = thermal_model

    def update_metrics(self, co2: float, price: float):
        energy = self.prev_load * 0.25 * 1e-3  # kWh
        temp_deviation = (
            abs(self.thermal_model.t_in - self.thermal_model.ac.default_setpoint) - 0.5
        )
        discomfort = max(temp_deviation - 1.0, 0) * 0.25  # degC-h, 0.5 deg tolerance
        emissions = energy * co2 * 1e-3  # kgCO2e (co2 is in gCO2eq_kWh)
        cost = energy * price * 1e-3  # $ (price in $/MWh)
        self.ep_energy_consumed += energy
        self.ep_co2_emission += emissions
        self.ep_energy_cost += cost
        self.ep_thermal_discomfort = discomfort
