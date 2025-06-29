from dataclasses import dataclass


@dataclass
class HysteresisCoolingSystemSpecification:
    max_cooling_power: float  # in watts
    cop: float  # watts/watts
    default_setpoint: float = 25.0  # deg C
    deadband: float = 0.5  # deg C
    # The following is a dynamic parameter which tracks if the cooler is on or off
    status: bool = False

    def get_cooling_power(
        self, t_in: float, t_set: float, t_amb: float
    ) -> tuple[float, float]:
        """
        Returns the cooling power and electric power consumption based on indoor temperature,
        setpoint, and hysteresis-based compressor control.

        Parameters:
            t_in (float): Indoor air temperature [°C]
            t_set (float): Desired setpoint temperature [°C]

        Returns:
            (cooling_power, electric_power): Tuple of cooling power [W] and electric power [W]
        """
        # Hysteresis controller
        # See if the cooler status needs to change
        if self.status and (t_in <= (t_set - self.deadband)):
            self.status = False
        if not self.status and (t_in >= (t_set + self.deadband)):
            self.status = True
        # The controller power should depend on the difference between t_in and t_set
        cooling_power = self.max_cooling_power if self.status else 0.0
        electric_power = abs(cooling_power / self.cop)
        return cooling_power, electric_power

    def reset(self):
        self.status = False
