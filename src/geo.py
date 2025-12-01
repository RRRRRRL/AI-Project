from typing import Tuple
import numpy as np
import pymap3d as pm

# WGS84 constants handled by pymap3d

def geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    # lat/lon in degrees, alt in meters
    # returns east, north, up in meters
    e, n, u = pm.geodetic2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt, deg=True)
    return np.asarray(e), np.asarray(n), np.asarray(u)

def enu_to_geodetic(e, n, u, ref_lat, ref_lon, ref_alt):
    lat, lon, alt = pm.enu2geodetic(e, n, u, ref_lat, ref_lon, ref_alt, deg=True)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt)