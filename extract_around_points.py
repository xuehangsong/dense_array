import h5py as h5
import numpy as np
from datetime import datetime, timedelta


def batch_delta_to_time(origin, x, time_format, delta_format):
    nx = len(x)
    y = []
    for ix in range(nx):
        if delta_format == "hours":
            temp_y = origin + timedelta(hours=x[ix])
        elif delta_format == "days":
            temp_y = origin + timedelta(days=x[ix])
        elif delta_format == "minutes":
            temp_y = origin + timedelta(minutes=x[ix])
        elif delta_format == "weeks":
            temp_y = origin + timedelta(weeks=x[ix])
        elif delta_format == "seconds":
            temp_y = origin + timedelta(seconds=x[ix])
        elif delta_format == "microseconds":
            temp_y = origin + timedelta(microseconds=x[ix])
        elif delta_format == "milliseconds":
            temp_y = origin + timedelta(milliseconds=x[ix])
        else:
            print("Sorry, this naive program only solve single time unit")
        y.append(temp_y.strftime(time_format))
    y = np.asarray(y)
    return(y)


def batch_time_to_delta(origin, x, time_format):
    nx = len(x)
    y = []
    for ix in range(nx):
        temp_y = abs(datetime.strptime(
            x[ix], time_format) - origin).total_seconds()
        y.append(temp_y)
    y = np.asarray(y)


#simu_dir = "/global/cscratch1/sd/kchen89/ERT_model/flux_estimation//"
simu_dir = "/global/cscratch1/sd/kchen89/ERT_model/flux_estimation/hete_4_2/"

date_origin = datetime.strptime("2015-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
date_start = datetime.strptime("2018-03-30 00:00:00", "%Y-%m-%d %H:%M:%S")
date_end = datetime.strptime("2018-05-02 00:00:00", "%Y-%m-%d %H:%M:%S")

date_start = datetime.strptime("2015-03-30 00:00:00", "%Y-%m-%d %H:%M:%S")
date_end = datetime.strptime("2015-05-02 00:00:00", "%Y-%m-%d %H:%M:%S")


h5_file = h5.File(simu_dir+"hete_4_2-19705.h5", "r")
h5_file = h5.File(simu_dir+"hete_4_2-20065.h5", "r")

x = h5_file["Coordinates"]["X [m]"][:]
y = h5_file["Coordinates"]["Y [m]"][:]
z = h5_file["Coordinates"]["Z [m]"][:]

dx = np.diff(x)
dy = np.diff(y)
dz = np.diff(z)

nx = len(dx)
ny = len(dy)
nz = len(dz)

ox = min(x)
oy = min(y)
oz = min(z)

ex = max(x)
ey = max(y)
ez = max(z)

x = x[0:nx] + 0.5 * dx
y = y[0:ny] + 0.5 * dy
z = z[0:nz] + 0.5 * dz

# h5_file.close()
times = list(h5_file.keys())[2:]
times_number = [float(x.split("Time:")[-1].split("h")[0]) if "h" in x else
                float(x.split("Time:")[-1].split("s")[0])/3600
                for x in times]
times_x = batch_delta_to_time(date_origin,
                              times_number,
                              "%Y-%m-%d %H:%M:%S",
                              "hours")
times_date = np.array([datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                       for x in times_x])


times_index = (times_date >= date_start)*(times_date <= date_end)
