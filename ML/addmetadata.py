import pandas as pd
from geopy.geocoders import Nominatim
import json
import urllib
import requests
import time
case_name = "mg_sizing_dataset_genome_1week_with_loc"
df = pd.read_csv("results/" + case_name + ".csv", sep=";|,", index_col='index', engine="python")
metadata = pd.read_csv("data/metadata.csv", sep=",",engine="python")
building_names = list(metadata.building_id)
print(metadata.loc[metadata.building_id==building_names[0], "sqm"][0])
for bname in building_names:
    if bname in list(df.name):
        df.loc[df.name == bname, 'area'] = list(metadata.loc[metadata.building_id==bname, "sqm"])[0]

df.to_csv("results/" + case_name + "_with_metadata.csv", sep=',',
          encoding='utf-8', index=True, decimal=".")

# for i in range(len(locations)):
#     time.sleep(3)
#     loc = locations[str(i)]["location"]
#     city = loc[:-3]
#     print(city)
#     if city == "Tanarive":
#         location = geolocator.geocode("Tananarive")
#     else:
#         location = geolocator.geocode(city)
#     # result = requests.get("https://api.opentopodata.org/v1/eudem25m?locations="+str(location.latitude)+","+str(location.longitude))
#     # elevation = result.json()["results"][0]["elevation"]
#     df.loc[df.location == loc, 'latitude'] = location.latitude
#     df.loc[df.location == loc, 'longitude'] = location.longitude
#     # df.loc[df.location == loc, 'elevation'] = elevation
