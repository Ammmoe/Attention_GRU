"""
coordinate_converter.py

Provides utility functions for converting geographic coordinates (latitude, longitude, altitude)
into metric Cartesian coordinates (meters) relative to a reference point.

This is useful for trajectory prediction, mapping, or robotics applications where
operations in meters are preferred over degrees.

Functions:
----------
latlon_to_meters(lat, lon, ref_lat=0.0, ref_lon=0.0)
    Convert latitude and longitude arrays (in degrees) into x and y coordinates in meters
    relative to a specified reference point. Altitude is preserved separately if needed.

Usage Example:
--------------
from coordinate_converter import latlon_to_meters

latitudes = [1.3521, 1.3530]
longitudes = [103.8198, 103.8205]
x, y = latlon_to_meters(latitudes, longitudes, ref_lat=1.3521, ref_lon=103.8198)
"""

import numpy as np


def latlon_to_meters(lat, lon, ref_lat=0.0, ref_lon=0.0):
    """
    Convert latitude and longitude coordinates to Cartesian meters relative to a reference point.

    Args:
        lat (array-like): Latitude values in degrees.
        lon (array-like): Longitude values in degrees.
        ref_lat (float, optional): Reference latitude in degrees (default=0.0).
        ref_lon (float, optional): Reference longitude in degrees (default=0.0).

    Returns:
        tuple of np.ndarray: (x, y) coordinates in meters relative to the reference point.

    Notes:
        - Uses a simple equirectangular approximation assuming small distances.
        - Earth radius is assumed to be 6,378,137 meters.
        - Altitude is not converted; only horizontal x/y coordinates are computed.
    """

    # Earth radius
    R = 6378137  # meters

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)
    return x, y
