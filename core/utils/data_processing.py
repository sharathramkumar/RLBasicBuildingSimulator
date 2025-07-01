import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from ..models.control_models import HysteresisCoolingSystemSpecification
from ..models.model_4r2c import ThermalModel4R2C, ModelParams4R2C
from ..testcase import BuildingTestcase


def extract_data(
    data_file_path: str,
    model_params_path: str,
    start_date: str,
    end_date: str,
    model_tag: str,
    return_testcase: bool = False,
):
    df = pd.read_pickle(data_file_path)
    with open(model_params_path) as ff:
        all_params_data = json.load(ff)
    this_model_params = all_params_data[model_tag]
    this_ac_params = this_model_params.pop("ac_details")
    cooling_spec = HysteresisCoolingSystemSpecification(
        this_ac_params["max_power"],
        cop=this_ac_params["cop"],
        default_setpoint=this_ac_params["preferred_setpoint"],
    )
    thermal_model = ThermalModel4R2C(
        ModelParams4R2C(**this_model_params),
        cooling_spec,
        init_t_in=25.0,
        init_t_e=30.0,
        init_t_ia=25.0,
        sim_step_size=0.25,
    )
    thermal_model.init_temperatures(df.Tamb_C.to_list(), df.Qirr_W_m2.to_list())
    df_sub = df[start_date:end_date]
    df_sub["Pelec_W"] = df_sub[f"{model_tag}_W"]
    df_sub = df_sub.drop(
        [x for x in df_sub.columns if x.startswith("Pelec") and x != "Pelec_W"], axis=1
    )
    if return_testcase:
        return generate_testcase(df_sub, thermal_model)
    return df_sub, thermal_model


def generate_testcase(df_sub: pd.DataFrame, thermal_model: ThermalModel4R2C):
    return BuildingTestcase(
        tamb_list=df_sub.Tamb_C.to_list(),
        qirr_list=df_sub.Qirr_W_m2.to_list(),
        pelec_list=df_sub.Pelec_W.to_list(),
        co2_list=df_sub.CO2_gCO2eq_kWh.to_list(),
        price_list=df_sub.USEP_SGD_MWh.to_list(),
        thermal_model=thermal_model,
    )


def plot_daily_trends(df: pd.DataFrame, ax: plt.Axes = None):
    """
    Plot daily mean and std trends of all columns in the given dataframe.
    Assumes datetime index at 15-minute resolution.
    """
    # Ensure datetime index and sort
    df = df.copy()
    df = df.sort_index()

    # Reshape each column into (num_days, 96)
    num_days = len(df) // 96
    daily_profiles = {
        col: df[col].values[: num_days * 96].reshape((num_days, 96))
        for col in df.columns
    }

    # Compute daily mean and std
    means = {col: data.mean(axis=0) for col, data in daily_profiles.items()}
    stds = {col: data.std(axis=0) for col, data in daily_profiles.items()}

    # Setup subplots
    if ax is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    else:
        axs = ax

    axs = axs.flatten()
    x = np.arange(96)  # Time steps per day

    title_map = {
        "Tamb_C": "Outdoor Temperature ($^o$C)",
        "Qirr_W_m2": "Solar Irradiance (W/m$^2^)",
        "CO2_gCO2eq_kWh": "Emissions Intensity (gCO$_2$eq/kWh)",
        "USEP_SGD_MWh": "Electricity Price (\$/MWh)",
    }

    for i, col in enumerate(df.columns):
        if col.startswith("Pelec"):
            continue
        mean = means[col]
        std = stds[col]
        axs[i].plot(x, mean, label="Mean")
        axs[i].fill_between(x, mean - std, mean + std, alpha=0.3, label="±1 Std")
        axs[i].set_title(title_map[col])
        axs[i].set_xlim([0, 95])
        if min(mean - std) < 0:
            axs[i].set_ylim([0, None])
        axs[i].set_xlabel("Timestep (15 min)")
        axs[i].set_ylabel(col)
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
