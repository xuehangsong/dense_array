# SUMMARY:      kriging_temperature.py
# USAGE:        use ert
# ORG:          Pacific Northwest National Laboratory
# AUTHOR:       Xuehang Song
# E-MAIL:       xuehang.song@pnnl.gov
# ORIG-DATE:    Mar-2020
# DESCRIPTION:
# DESCRIPTION-END


import numpy as np
from pykrige.uk3d import UniversalKriging3D
import multiprocessing as mp
from datetime import datetime, timedelta
from sklearn.externals import joblib

da1_joblib = "/mnt/e/dense_array/data/da1.joblib"
output_dir = "/mnt/e/dense_array/temp/"

da1 = joblib.load(da1_joblib)
# define dimension
south = 116255
north = 116315
west = 594455
east = 594490
bottom = 102
top = 106
# mesh
ox = west
ex = east
oy = south
ey = north
oz = bottom
ez = top
dx = 0.5
dy = 0.5
dz = 0.1
dx = np.array([dx]*int((ex-ox)/dx))
dy = np.array([dy]*int((ey-oy)/dy))
dz = np.array([dz]*int((ez-oz)/dz))
x = ox+np.cumsum(dx) - 0.5*dx
y = oy+np.cumsum(dy) - 0.5*dy
z = oz+np.cumsum(dz) - 0.5*dz
nx = len(dx)
ny = len(dy)
nz = len(dz)
material_z = np.append(np.arange(z[-1], 96, -0.1)[::-1], z)
material_nz = len(material_z)
bath_x = np.arange(594300.5598+0.5, 594600.5598, 1)
bath_y = np.arange(116000.6771+0.5, 116500.6771, 1)
date_origin = datetime.strptime("2017-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

thermistors = [x for x in list(
    da1.keys()) if x != "time" and x != "delta_day"]
ntherm = len(thermistors)
therm_x = np.array([da1[x]["easting"] for x in thermistors])
therm_y = np.array([da1[x]["northing"] for x in thermistors])
therm_z = np.array([da1[x]["elevation"] for x in thermistors])
therm_data = np.array([da1[x]["temperature"] for x in thermistors])

# choose time segments (>30 days) with all thermistors functional
valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
valid_index = np.where(valid_ntherm > 30)[0]


# for t_index in valid_index:


def universal_kriging(t_index):
    working_therm = ~np.isnan(therm_data[:, t_index])
    uk3d = UniversalKriging3D(
        therm_x[working_therm],
        therm_y[working_therm],
        therm_z[working_therm],
        therm_data[working_therm, t_index],
        variogram_model='linear',
        drift_terms=['regional_linear'])
    k3d, ss3d = uk3d.execute('grid', x, y, z)
    # output data to hdf5
    temp_data = k3d.data.swapaxes(0, 2).flatten(order="F")
    joblib.dump(temp_data, output_dir+str(t_index)+".joblib")


ncore = 4
pool = mp.Pool(processes=ncore)
pool.map(universal_kriging, valid_index)
