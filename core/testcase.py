from .agent import ControlAgent
from .models.model_4r2c import ThermalModel4R2C
import matplotlib.pyplot as plt
import numpy as np


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
        self.preferred_temp = self.thermal_model.ac.default_setpoint  # deg C

    def run_testcase(self, agent: ControlAgent):
        prev_tot_elec = 0.0
        self.reset()

        for t in range(len(self.tamb_list)):
            tamb = self.tamb_list[t]
            qirr = self.qirr_list[t]
            pelec = self.pelec_list[t]
            co2 = self.co2_list[t]
            price = self.price_list[t]

            # Agent decides setpoint
            setpoint = agent.get_ac_setpoint(
                time=(t % 96) / 96.0,
                t_in=self.thermal_model.t_in,
                t_amb=tamb,
                q_irr=qirr,
                prev_load=prev_tot_elec,
                curr_price=price,
                curr_co2=co2,
            )

            # Update thermal model
            ac_cooling_power = self.thermal_model.step(setpoint, tamb, qirr, pelec)

            # Track metrics
            energy = (pelec + ac_cooling_power) * 0.25 * 1e-3  # kWh
            temp_deviation = abs(self.thermal_model.t_in - self.preferred_temp) - 0.5
            discomfort = (
                max(temp_deviation - 1.0, 0) * 0.25
            )  # degC-h, 1.0 deg tolerance
            emissions = energy * co2 * 1e-3  # kgCO2e (co2 is in gCO2eq_kWh)
            cost = energy * price * 1e-3  # $ (price in $/MWh)

            self.total_energy += energy
            self.total_discomfort += discomfort
            self.total_emissions += emissions
            self.total_cost += cost

            prev_tot_elec = pelec + ac_cooling_power

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

    def plot_data(self, start: int, end: int):
        """
        Plot simulation results between two time indices using a 5-row subplot.

        Args:
            start (int): Start index (inclusive)
            end (int): End index (exclusive)
        """
        if not hasattr(self.thermal_model, "history"):
            raise RuntimeError(
                "Thermal model has no recorded history. Did you run the testcase?"
            )

        hist = self.thermal_model.history
        idx_range = slice(start, end)
        x = np.arange(start, end) / 4.0  # convert to hours

        fig, axs = plt.subplots(5, 1, figsize=(6, 10), sharex=True)

        # 1. Outdoor temperature
        axs[0].plot(
            x, self.tamb_list[idx_range], label="Outdoor Temp (°C)", color="orange"
        )
        axs[0].set_ylabel("Tamb (°C)")
        axs[0].legend()
        axs[0].grid(True)

        # 2. Power: AC + non-AC
        p_ac = np.array(hist.p_ac[idx_range])
        p_other = np.array(self.pelec_list[idx_range])
        axs[1].plot(x, p_other + p_ac, label="Total Power (W)", color="purple")
        axs[1].plot(x, p_ac, label="AC Power (W)", color="red", linestyle="--")
        axs[1].set_ylabel("Power (W)")
        axs[1].legend()
        axs[1].grid(True)

        # 3. Indoor temp with control region and actions
        axs[2].axhspan(23, 25, color="green", alpha=0.1, label="Comfort Zone")
        axs[2].plot(x, hist.t_in[idx_range], label="Indoor Temp (°C)", color="black")
        axs[2].plot(
            x,
            hist.t_set[idx_range],
            label="Setpoint (°C)",
            color="blue",
            linestyle="--",
        )
        axs[2].axhline(
            y=self.thermal_model.ac.default_setpoint, color="grey", linestyle=":"
        )
        axs[2].set_ylabel("T_in / Setpoint (°C)")
        axs[2].legend()
        axs[2].grid(True)

        # 4. CO2 signal
        axs[3].plot(
            x, self.co2_list[idx_range], label="Grid CO₂ Intensity", color="brown"
        )
        axs[3].set_ylabel("gCO₂/kWh")
        axs[3].legend()
        axs[3].grid(True)

        # 5. Price signal
        axs[4].plot(
            x, self.price_list[idx_range], label="USEP Price ($/MWh)", color="teal"
        )
        axs[4].set_ylabel("Price")
        axs[4].set_xlabel("Time (hours)")
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout()
        plt.show()
