def calc_step_metrics_from_state(prev_state, action, curr_state):
    # This function returns the energy consumed, the co2 emissions, the step cost
    energy = curr_state["prev_load"] * 0.25 * 1e-3  # kWh
    co2 = prev_state["curr_co2"]
    price = prev_state["curr_price"]
    emissions = energy * co2 * 1e-3  # kgCO2e (co2 is in gCO2eq_kWh)
    cost = energy * price * 1e-3  # $ (price in $/MWh)
    temp_deviation = abs(curr_state["t_in"] - 25.0) - 0.5
    discomfort = max(temp_deviation - 1.0, 0) * 0.25  # degC-h, 1 deg tolerance
    return discomfort, energy, emissions, cost
