import pandas as pd
import json
import os
from sizer import Sizer
from representative_days import RepresentativeDays, create_rd_output_path
from sizing_configuration import SizingConfiguration

rd = RepresentativeDays()
simple_sizing = False

if simple_sizing:
    with open("data/ANBRIMEX2/hs_AMBRIMEX.json", 'r') as file:
        microgrid = json.load(file)
    config = SizingConfiguration(microgrid["parameters"])
    case_name = config.case_name
    initial_state = {"state_of_charge": {storage["name"]: 0.2 for storage in microgrid["storages"]}}
    # initial_state = {"state_of_charge": {storage["name"]: 50 for storage in microgrid["storages"]}}
    time_series = pd.read_csv("data/ANBRIMEX2/hs_0_data_15min_ANBRIMEX.csv", parse_dates=True, index_col='DateTime')
    rd_output_path = create_rd_output_path("results/" + case_name)
    selected_days, extracted_series, extracted_weights = rd.get_representative_days(time_series, config,
                                                                                    output_path=rd_output_path)
    sizer = Sizer(microgrid, time_series, config, initial_state)
    sizer.size(selected_days, extracted_series, extracted_weights)
else:
    time_series = pd.read_csv("data/timeseries.csv", sep=";", parse_dates=True, index_col='DateTime')
    pv_production = pd.read_csv("data/pv_production_location.csv", sep=";", parse_dates=True, index_col='DateTime')
    load_data = pd.read_csv("data/load_genome.csv", sep=";|,", parse_dates=True, engine="python", index_col='DateTime')
    with open("data/locations.json", 'rb') as file:
        locations = json.load(file)

    companies = load_data.columns.to_list()
    with open("data/genome.json", 'r') as file:
        microgrid = json.load(file)

    config = SizingConfiguration(microgrid["parameters"])
    # peak = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
    #peak = [1, 10, 100, 1000]
    keys = ["name", "location", "longitude", "latitude", "peak_load", "avg_peak_winter",
            "avg_peak_spring", "avg_peak_summer", "avg_peak_autumn", "avg_base_winter",
            "avg_base_spring", "avg_base_summer", "avg_base_autumn", "purchase_price",
            "off-grid", "PV", "BAT", "RBAT", "INV", "GEN", "NPV"]
    db_ml = {k: [] for k in keys}
    for c in companies:
        config.case_name = c
        case_name = config.case_name
        initial_state = {"state_of_charge": {storage["name"]: 50 for storage in microgrid["storages"]}}
        # for l in range(len(peak)):  # to replace by len(l_p)
        time_series["Load"] = load_data[case_name]
        peak = load_data[case_name].max()
        microgrid["non_flexible_loads"][0]["capacity"] = peak

        # determine the connection level based on the peak load
        if 0 < peak <= 100:
            microgrid['non_flexible_loads'][0]["connection_type"] = 'BT'
        elif 100 < peak <= 250:
            microgrid['non_flexible_loads'][0]["connection_type"] = 'TBT'
        elif 250 < peak <= 5000:
            microgrid['non_flexible_loads'][0]["connection_type"] = 'MT'
        else:
            microgrid['non_flexible_loads'][0]["connection_type"] = 'TMT'

        # determine the purchase price based on the total annual consumption
        sim_time_step_hour = config.simulation_step / 60.0
        annual_consumption = time_series["Load"].sum() * peak * sim_time_step_hour

        # for a consumption <= 10000 kWh/y -->  0.23 €/kWh
        max_pp = 0.23
        cons_max_pp = 10000
        # for a consumption >= 2000000 kWh/y --> 0.1 €/kWh
        min_pp = 0.10
        conso_min_pp = 2000000

        if annual_consumption > conso_min_pp:
            purchase_price = min_pp
        elif annual_consumption < cons_max_pp:
            purchase_price = max_pp
        else:
            purchase_price = ((min_pp - max_pp) / (conso_min_pp - cons_max_pp)) * (
                    annual_consumption - cons_max_pp) + max_pp
        time_series["purchase_price"] = [purchase_price] * len(time_series)

        for i in range(len(locations)):  # locations
            print(c, locations[str(i)], peak)
            time_series["PV"] = pv_production[locations[str(i)]["location"]]
            rd_output_path = create_rd_output_path("results/" + case_name)
            selected_days, extracted_series, extracted_weights = rd.get_representative_days(time_series, config,
                                                                                            output_path=rd_output_path)
            config.grid_tied = False
            config.grid_connection_cost = False
            sizer = Sizer(microgrid, time_series, config, initial_state)
            sizer.size(selected_days, extracted_series, extracted_weights)
            # Load data features
            load_data['Month'] = load_data.index.map(lambda x: x.month)
            load_data['Season'] = load_data.index.map(lambda x: x.month % 12 // 3 + 1)

            # Latitude, longitude, elevation
            loc = locations[str(i)]["location"]
            city = loc[:-3]
            # geo_location = geolocator.geocode(city)
            # query = ('https://api.open-elevation.com/api/v1/lookup'
            #          f'?locations={geo_location.latitude},{geo_location.longitude}')
            # r = requests.get(query).json()  # json object, various ways you can extract value
            # # one approach is to use pandas json functionality:
            # elevation = pd.io.json.json_normalize(r, 'results')['elevation'].values[0]

            db_ml["name"].append(case_name)
            db_ml["longitude"].append(0)
            db_ml["latitude"].append(0)
            # db_ml["elevation"].append(elevation)
            db_ml["peak_load"].append(peak)
            db_ml["avg_base_winter"].append(load_data.loc[load_data['Season'] == 1].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_spring"].append(load_data.loc[load_data['Season'] == 2].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_summer"].append(load_data.loc[load_data['Season'] == 3].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_autumn"].append(load_data.loc[load_data['Season'] == 4].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_winter"].append(load_data.loc[load_data['Season'] == 1].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_spring"].append(load_data.loc[load_data['Season'] == 2].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_summer"].append(load_data.loc[load_data['Season'] == 3].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_autumn"].append(load_data.loc[load_data['Season'] == 4].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["location"].append(city)
            db_ml["purchase_price"].append(purchase_price)
            db_ml["off-grid"].append(int(not config.grid_tied))
            db_ml["PV"].append(sizer.op_sizing["PV"][0])
            db_ml["BAT"].append(sizer.op_sizing["BAT"][0])
            db_ml["INV"].append(sizer.op_sizing["INV"][0])
            if not sizer.op_sizing["GEN"]:
                db_ml["GEN"].append(0)
            else:
                db_ml["GEN"].append(sizer.op_sizing["GEN"][0])
            db_ml["RBAT"].append(sizer.op_sizing["BATRVST"])
            db_ml["NPV"].append(sizer.op_sizing["NPV"])

            # size with grid tied and grid connection cost modes
            config.grid_tied = True
            config.grid_connection_cost = True

            sizer = Sizer(microgrid, time_series, config, initial_state)
            sizer.size(selected_days, extracted_series, extracted_weights)
            db_ml["name"].append(case_name)
            db_ml["longitude"].append(0)
            db_ml["latitude"].append(0)
            # db_ml["elevation"].append(elevation)
            db_ml["peak_load"].append(peak)
            db_ml["avg_base_winter"].append(load_data.loc[load_data['Season'] == 1].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_spring"].append(load_data.loc[load_data['Season'] == 2].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_summer"].append(load_data.loc[load_data['Season'] == 3].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_base_autumn"].append(load_data.loc[load_data['Season'] == 4].nsmallest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_winter"].append(load_data.loc[load_data['Season'] == 1].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_spring"].append(load_data.loc[load_data['Season'] == 2].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_summer"].append(load_data.loc[load_data['Season'] == 3].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["avg_peak_autumn"].append(load_data.loc[load_data['Season'] == 4].nlargest(
                int(len(load_data.loc[load_data['Season'] == 1]) * 0.2), [case_name])[
                                                case_name].mean())
            db_ml["location"].append(city)
            db_ml["purchase_price"].append(purchase_price)
            db_ml["off-grid"].append(int(not config.grid_tied))
            db_ml["PV"].append(sizer.op_sizing["PV"][0])
            db_ml["BAT"].append(sizer.op_sizing["BAT"][0])
            db_ml["INV"].append(sizer.op_sizing["INV"][0])
            if not sizer.op_sizing["GEN"]:
                db_ml["GEN"].append(0)
            else:
                db_ml["GEN"].append(sizer.op_sizing["GEN"][0])
            db_ml["RBAT"].append(sizer.op_sizing["BATRVST"])
            db_ml["NPV"].append(sizer.op_sizing["NPV"])

            db_ml_df = pd.DataFrame(db_ml)
            db_ml_df.index.name = "index"

            if not os.path.exists("results/"):
                os.makedirs("results/")
            db_ml_df.to_csv("results/mg_sizing_dataset_genome.csv", sep=',',
                            encoding='utf-8', index=True, decimal=".")

