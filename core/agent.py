from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from .utils.scaler import ScalerManager
import json
import numpy as np


class ControlAgent(ABC):
    @abstractmethod
    def get_ac_setpoint(
        self,
        time: float,
        t_in: float,
        t_amb: float,
        q_irr: float,
        prev_load: float,
        curr_price: float,
        curr_co2: float,
    ) -> float:
        """
        Decide the AC setpoint temperature based on current conditions.

        Parameters:
            time (float): time of the day between 0.0 (12 AM) and 1.0 (11:59:59 PM)
            t_in (float): Current indoor temperature [°C]
            t_amb (float): Current outdoor/ambient temperature [°C]
            q_irr (float): Current solar irradiation [W/m²]
            prev_load (float): Previous timestep's electrical load [W]
            curr_price (float): Current electricity price [$/Wh]
            curr_co2 (float): Current grid CO₂ intensity [gCO₂e/Wh]

        Returns:
            float: Desired AC setpoint temperature [°C]
        """
        pass


class StaticAgent(ControlAgent):
    def __init__(self, setpoint=24.0):
        self.sp = setpoint

    def get_ac_setpoint(
        self, time, t_in, t_amb, q_irr, prev_load, curr_price, curr_co2
    ):
        return self.sp


class RLAgent(ControlAgent):
    def __init__(self, agent: PPO, agent_config_path: str, scaler_config_path: str):
        self.agent = agent
        self.scaler_manager = ScalerManager(scaler_config_path)
        with open(agent_config_path) as ff:
            self.agent_config = json.load(ff)
        self.action_type = self.agent_config["action_type"]
        self.state_keys = self.agent_config["state_keys"]
        self.action_space_config = self.agent_config["action_space"]

    def get_ac_setpoint(
        self, time, t_in, t_amb, q_irr, prev_load, curr_price, curr_co2
    ):
        obs = self._get_obs(time, t_in, t_amb, q_irr, prev_load, curr_price, curr_co2)
        action = self.agent.predict(obs, deterministic=True)[0].item()
        if self.action_type == "discrete":
            action = np.array([self.action_space_config["values"][action]])
            act_min, act_max = (
                self.action_space_config["values"][0],
                self.action_space_config["values"][-1],
            )
        else:
            act_min, act_max = self.action_space_config["range"]
        act = (action + 1) / 2 * (act_max - act_min) + act_min
        return act

    def _get_obs(self, time, t_in, t_amb, q_irr, prev_load, curr_price, curr_co2):
        state_dict = {
            "time": time,
            "t_in": t_in,
            "t_amb": t_amb,
            "q_irr": q_irr,
            "prev_load": prev_load,
            "curr_price": curr_price,
            "curr_co2": curr_co2,
        }
        obs = [
            self.scaler_manager.scale(key, state_dict[key]) for key in self.state_keys
        ]
        return np.array(obs, dtype=np.float32)
