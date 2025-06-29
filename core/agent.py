from abc import ABC, abstractmethod


class ControlAgent(ABC):
    @abstractmethod
    def get_ac_setpoint(
        self,
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

    def get_ac_setpoint(self, t_in, t_amb, q_irr, prev_load, curr_price, curr_co2):
        return self.sp
