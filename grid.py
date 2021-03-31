from device import Device
from capex import Capex
class Grid:
    def __init__(self, grid_data):
        self.storages = [Device(s) for s in grid_data["storages"]]
        self.h2_storages = [Device(s) for s in grid_data["h2_storages"]]
        self.h2_tanks = [Device(t) for t in grid_data["h2_tanks"]]
        self.flexible_loads = [Device(l) for l in grid_data["flexible_loads"]]
        self.non_flexible_loads = [Device(l) for l in grid_data["non_flexible_loads"]]
        self.inverters = [Device(i) for i in grid_data["inverters"]]
        self.non_steerable_generators = [Device(g) for g in grid_data["non_steerable_generators"]]
        self.sheddable_loads = [Device(l) for l in grid_data["sheddable_loads"]]
        self.steerable_generators = [Device(g) for g in grid_data["steerable_generators"]]
        self.pv_capex = [Capex(c) for c in grid_data["pv_capex"]]
        self.inverter_capex = [Capex(c) for c in grid_data["inverter_capex"]]
        self.battery_capex = [Capex(c) for c in grid_data["battery_capex"]]
        storages_name = [s.name for s in self.storages]
        h2_storages_name = [s.name for s in self.h2_storages]
        h2_tanks_name = [s.name for s in self.h2_tanks]
        flexible_loads_name = [s.name for s in self.flexible_loads]
        non_flexible_loads_name = [s.name for s in self.non_flexible_loads]
        inverters_name = [s.name for s in self.inverters]
        non_steerable_generators_name = [s.name for s in self.non_steerable_generators]
        sheddable_loads_name = [s.name for s in self.sheddable_loads]
        steerable_generators_name = [s.name for s in self.steerable_generators]
        self.devices_names = storages_name + h2_storages_name + h2_tanks_name + flexible_loads_name + non_flexible_loads_name + inverters_name + non_steerable_generators_name + sheddable_loads_name + steerable_generators_name
        


