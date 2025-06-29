import pandas as pd
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
    if return_testcase:
        return generate_testcase(df_sub, thermal_model, model_tag)
    return df_sub, thermal_model


def generate_testcase(
    df_sub: pd.DataFrame, thermal_model: ThermalModel4R2C, model_tag: str
):
    return BuildingTestcase(
        tamb_list=df_sub.Tamb_C.to_list(),
        qirr_list=df_sub.Qirr_W_m2.to_list(),
        pelec_list=df_sub[f"{model_tag}_W"].to_list(),
        co2_list=df_sub.CO2_gCO2eq_kWh.to_list(),
        price_list=df_sub.USEP_SGD_MWh.to_list(),
        thermal_model=thermal_model,
    )
