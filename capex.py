class Capex:
    def __init__(self, capex_data):
        for key in capex_data.keys():
            setattr(self, key, capex_data[key])

    def __str__(self):
        #import json
        return self.name