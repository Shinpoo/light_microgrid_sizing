import pandas as pd
from geopy.geocoders import Nominatim
import json
import urllib
import requests
import time
case_name = "mg_sizing_dataset_genome_4week"
df = pd.read_csv("results/" + case_name + ".csv", sep=";|,", index_col='index', engine="python")
geolocator = Nominatim(user_agent="light_microgrid_sizing")
# result = requests.get("https://api.opentopodata.org/v1/eudem25m?locations=51.875127,-3.341298")
# print(result.json()["results"][0]["elevation"])
with open("data/locations.json", 'rb') as file:
    locations = json.load(file)
for i in range(len(locations)):
    # time.sleep(3)
    loc = locations[str(i)]["location"]
    city = loc[:-3]
    print(city)
    if city == "Tanarive":
        location = geolocator.geocode("Tananarive")
    else:
        location = geolocator.geocode(city)
    # result = requests.get("https://api.opentopodata.org/v1/eudem25m?locations="+str(location.latitude)+","+str(location.longitude))
    # elevation = result.json()["results"][0]["elevation"]
    time.sleep(10)
    df.loc[df.location == city, 'latitude'] = location.latitude
    df.loc[df.location == city, 'longitude'] = location.longitude
    # df.loc[df.location == loc, 'elevation'] = elevation
df.to_csv("results/" + case_name + "_with_loc.csv", sep=';',
          encoding='utf-8', index=True, decimal=".")
