import pandas as pd
import json
import os
from sizer import Sizer
import shutil
from representative_days import RepresentativeDays, create_rd_output_path
from sizing_configuration import SizingConfiguration
#Select representative days

#Get the data
pv_production = pd.read_csv("data/pv_production_location.csv", sep=";", parse_dates=True, index_col='DateTime')
load_data = pd.read_csv("data/load.csv", sep=";", parse_dates=True, index_col='DateTime')
time_series = pd.read_csv("data/timeseries.csv", sep=";", parse_dates=True, index_col='DateTime')
with open("data/locations.json", 'rb') as file:
    locations = json.load(file)

with open("data/ANBRIMEX/ANBRIMEX.json", 'r') as file:
    microgrid = json.load(file)

config = SizingConfiguration(microgrid["parameters"])
case_name = config.case_name

#Representative days
rd = RepresentativeDays()


initial_state = {"state_of_charge": {storage["name"]: 50 for storage in microgrid["storages"]}}
sizing_db = {"name": case_name, "l_p * 1": [], "l_p * 5": [], "l_p * 10": [],
            "l_p * 25": [], "l_p * 50": [], "l_p * 100": [], "l_p * 250": [], "l_p * 500": [],
            "l_p * 1000": [], "l_p * 5000": []}
l_p = ["l_p * 1", "l_p * 5", "l_p * 10", "l_p * 25", "l_p * 50", "l_p * 100", "l_p * 250", "l_p * 500",
        "l_p * 1000", "l_p * 5000"]
peak = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
for l in range(len(l_p)):
    for i in range(len(locations)):
        sizing_db[l_p[l]].append(
            {"location": locations[str(i)]["location"], "purchase_price": 0.0, "sizing_results": {
                "grid_tied": {"peak_load": peak[l], "PV": 0.0, "INV": 0.0, "BAT": 0.0, "RBAT": 0.0, "GEN": 0.0, "NPV": 0.0},
                "off_grid": {"peak_load": peak[l], "PV": 0.0, "INV": 0.0, "BAT": 0.0, "RBAT": 0.0, "GEN": 0.0, "NPV": 0.0}}})


for l in range(1):#len(l_p)
    time_series["Load"] = load_data[case_name]
    microgrid["non_flexible_loads"][0]["capacity"] = peak[l]

    # determine the connection level based on the peak load
    if 0 < peak[l] <= 100:
        microgrid['non_flexible_loads'][0]["connection_type"] = 'BT'
    elif 100 < peak[l] <= 250:
        microgrid['non_flexible_loads'][0]["connection_type"] = 'TBT'
    elif 250 < peak[l] <= 5000:
        microgrid['non_flexible_loads'][0]["connection_type"] = 'MT'
    else:
        microgrid['non_flexible_loads'][0]["connection_type"] = 'TMT'

    # determine the purchase price based on the total annual consumption
    sim_time_step_hour = config.simulation_step/60.0
    annual_consumption = time_series["Load"].sum() * peak[l] * sim_time_step_hour

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

    
    for i in range(1):#len(locations)
        sizing_db[l_p[l]][i]["purchase_price"] = purchase_price
        time_series["PV"] = pv_production[locations[str(i)]["location"]]
        rd_output_path = create_rd_output_path("results/" + case_name)
        selected_days, extracted_series, extracted_weights = rd.get_representative_days(time_series, config, output_path=rd_output_path)
        sizer = Sizer(microgrid, time_series, config, initial_state)
        sizer.size(selected_days, extracted_series, extracted_weights)

        #selected_days, extracted_series, extracted_weights = sizer.get_representative_days(time_series, config, output_path)

        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["PV"] = sizer.op_sizing["PV"]
        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["BAT"] = sizer.op_sizing["BAT"]
        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["RBAT"] = sizer.op_sizing["BATRVST"]
        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["INV"] = sizer.op_sizing["INV"]
        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["GEN"] = sizer.op_sizing["GEN"]
        # sizing_db[l_p[l]][i]["sizing_results"]['off_grid']["NPV"] = sizer.op_sizing["NPV"]

        # size with grid tied and grid connection cost modes
        # sizing_config.grid_tied = True
        # sizing_config.grid_connection_cost = True

        # sizer = Sizer(microgrid, time_series, sizing_config, initial_state)
        # sizer.size()

        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["PV"] = sizer.op_sizing["PV"]
        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["BAT"] = sizer.op_sizing["BAT"]
        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["RBAT"] = sizer.op_sizing["BATRVST"]
        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["INV"] = sizer.op_sizing["INV"]
        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["GEN"] = sizer.op_sizing["GEN"]
        # sizing_db[l_p[l]][i]["sizing_results"]['grid_tied']["NPV"] = sizer.op_sizing["NPV"]

        path = "results/" + case_name + ".json"
        if not os.path.exists("results/"):
            os.makedirs("results/")
        with open(path, 'w') as file:
            json.dump(sizing_db, file, indent=4, separators=(',', ': '))
#Run opt problem
#results