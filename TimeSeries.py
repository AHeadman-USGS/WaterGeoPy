import netCDF4, os
import pandas as pd
import datetime as dt

def near(array, value):
    """
    array: 2d Array of values taken from daymet NetCDF input file.
    value: Input value derived from user provided shapefile.
    idx: Output for actual value nearest to user provided value.
    """

    # Helper function for build_prcp/build temps.  Finds the actual  nearest x, y coordinate in the matrix
    # to the user input coordinate.
    idx = (abs(array - value)).argmin()
    return idx

def build_prcp(f, x, y):
    """
    This needs a docstring!
    """

    # Read in and build the netCDF4 parameters
    nc = netCDF4.Dataset(f)
    lat = nc.variables['y'][:]
    lon = nc.variables['x'][:]
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)

    # Building the indexes points.
    # By default this starts when Daymet starts, though this could be flexible.
    # Currently, this only accepts Daymet data.
    start = dt.datetime(1980, 1, 1)
    end = dt.datetime.utcnow()
    istart = netCDF4.date2index(start, time_var, select='nearest')
    istop = netCDF4.date2index(end, time_var, select='nearest')
    lati = y
    loni = x
    ix = near(lon, loni)
    iy = near(lat, lati)

    # Selecting the variables.
    prcp = nc.variables['prcp'][:]
    hs = prcp[0, istart:istop, ix, iy]
    tim = dtime[istart:istop]

    # Arranging data into pandas df.
    prcp_ts = pd.Series(hs, index=tim, name='precipitation (mm/day)')
    prcp_ts = pd.DataFrame(prcp_ts)
    prcp_ts.reset_index(inplace=True)
    prcp_ts.columns = ['Index', 'precipitation (mm/day)']
    prcp_ts['date'] = prcp_ts['Index']
    prcp_ts.set_index('Index', drop=True, inplace=True)

    return prcp_ts


def build_temps(f, x, y):
    """
    This also needs a docstring.
    """

    # Read in and build the netCDF4 parameters
    nc = netCDF4.Dataset(f)
    lat = nc.variables['y'][:]
    lon = nc.variables['x'][:]
    time_var = nc.variables['time']
    dtime = netCDF4.num2date(time_var[:], time_var.units)

    # Building the indexes points.
    # By default this starts when Daymet starts, though this could be flexible.
    # Currently, this only accepts Daymet data.
    start = dt.datetime(1980, 1, 1)
    end = dt.datetime.utcnow()
    istart = netCDF4.date2index(start, time_var, select='nearest')
    istop = netCDF4.date2index(end, time_var, select='nearest')
    lati = y
    loni = x
    ix = near(lon, loni)
    iy = near(lat, lati)

    # Selecting/subsetting the NetCDF dataset.
    temps = nc.variables['tmax'][:]
    hs = temps[0, istart:istop, ix, iy]
    tim = dtime[istart:istop]

    # Arranging data into pandas df.
    temps_ts = pd.Series(hs, index=tim, name='temperature (celsius)')
    temps_ts = pd.DataFrame(temps_ts)
    temps_ts.reset_index(inplace=True)
    temps_ts.columns = ['Index', 'temperature (celsius)']
    temps_ts['date'] = temps_ts['Index']
    temps_ts.set_index('Index', drop=True, inplace=True)

    return temps_ts
