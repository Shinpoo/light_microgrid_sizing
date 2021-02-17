class SizingConfiguration:
    def __init__(self, config_data): 
        for key in config_data.keys():    
            setattr(self, key, config_data[key])