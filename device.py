class Device:
    def __init__(self, device_data): 
        for key in device_data.keys():    
            setattr(self, key, device_data[key])