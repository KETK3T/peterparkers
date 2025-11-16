'''
TODO: Load the data

'''

class GPS:
    def __init__(self, rate_hz):
        self.latitude = None
        self.longitude = None
        self.altitude = None

    def __str__(self):
        return (
            f"Latitude: {self.latitude:.3f}\n"
            f"Longitude: {self.longitude:.3f}\n"
            f"Altitude: {self.altitude:.3f}\n"
        )

    def get_lat(self):
        return self.latitude
    
    def get long(self):
        return self.longitude
    
    def get alti(self):
        return self.altitude
    
    