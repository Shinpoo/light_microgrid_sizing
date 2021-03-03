import pandas as pd
from geopy.geocoders import Nominatim
import json

case_name = "ANBRIMEX"
df = pd.read_csv("results/" + case_name + ".csv", sep=";", index_col='index')
geolocator = Nominatim(user_agent="light_microgrid_sizing")

with open("data/locations.json", 'rb') as file:
    locations = json.load(file)
for i in range(len(locations)):
    loc = locations[str(i)]["location"]
    city = loc[:-3]
    print(city)
    if city == "Tanarive":
        location = geolocator.geocode("Tananarive")
    else:
        location = geolocator.geocode(city)
    df.loc[df.location == loc, 'latitude'] = location.latitude
    df.loc[df.location == loc, 'longitude'] = location.longitude
df.to_csv("results/" + case_name + "_with_loc.csv", sep=';',
          encoding='utf-8', index=True, decimal=".")
