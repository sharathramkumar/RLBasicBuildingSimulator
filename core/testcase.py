from .agent import ControlAgent
from .models.model_4r2c import ThermalModel4R2C


class BuildingTestcase:
    def __init__(
        self,
        tamb_list: list,
        qirr_list: list,
        pelec_list: list,
        co2_list: list,
        price_list: list,
        thermal_model: ThermalModel4R2C,
    ):
        # Check input lengths
        n = len(tamb_list)
        if not all(
            len(lst) == n for lst in [qirr_list, pelec_list, co2_list, price_list]
        ):
            raise ValueError("All input lists must be of the same length.")

        self.tamb_list = tamb_list
        self.qirr_list = qirr_list
        self.pelec_list = pelec_list
        self.co2_list = co2_list
        self.price_list = price_list
        self.thermal_model = thermal_model

        # Metrics
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.total_emissions = 0.0
        self.total_discomfort = 0.0

        # Default preferred indoor temperature
        self.preferred_temp = 24.0  # deg C

    def run_testcase(self, agent: ControlAgent):
        prev_pelec = 0.0
        self.reset()

        for t in range(len(self.tamb_list)):
            tamb = self.tamb_list[t]
            qirr = self.qirr_list[t]
            pelec = self.pelec_list[t]
            co2 = self.co2_list[t]
            price = self.price_list[t]

            # Agent decides setpoint
            setpoint = agent.get_ac_setpoint(
                t_in=self.thermal_model.t_in,
                t_amb=tamb,
                q_irr=qirr,
                prev_load=prev_pelec,
                curr_price=price,
                curr_co2=co2,
            )

            # Update thermal model
            ac_cooling_power = self.thermal_model.step(setpoint, tamb, qirr)

            # Track metrics
            energy = (pelec + ac_cooling_power) * 0.25 * 1e-3  # kWh
            discomfort = (
                abs(self.thermal_model.t_in - self.preferred_temp) * 0.25
            )  # degC-h
            emissions = energy * co2 * 1e-3  # kgCO2e (co2 is in gCO2eq_kWh)
            cost = energy * price * 1e-3  # $ (price in $/MWh)

            self.total_energy += energy
            self.total_discomfort += discomfort
            self.total_emissions += emissions
            self.total_cost += cost

            prev_pelec = pelec

    def show_metrics(self):
        print("=== Testcase Metrics ===")
        print(f"Total energy (kWh): {self.total_energy:.2f}")
        print(f"Total discomfort (degree-hours): {self.total_discomfort:.2f}")
        print(f"Total cost ($): {self.total_cost:.2f}")
        print(f"Total emissions (kgCO2e): {self.total_emissions:.2f}")

    def reset(self):
        # Metrics
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.total_emissions = 0.0
        self.total_discomfort = 0.0

        # Models
        self.thermal_model.reset()
