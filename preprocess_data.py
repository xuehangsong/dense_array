# SUMMARY:      preprocess_data.py
# USAGE:        main scripts to do the rtd analyze
# ORG:          Pacific Northwest National Laboratory
# AUTHOR:       Xuehang Song
# E-MAIL:       xuehang.song@pnnl.gov
# ORIG-DATE:    Mar-2020
# DESCRIPTION:
# DESCRIPTION-END

from datetime import datetime, timedelta
import numpy as np
import csv
from sklearn.externals import joblib
from datetime import datetime, timedelta

data_dir = "/mnt/e/dense_array/data/"
coord_file = data_dir+"DA1_CRiver_TM_T_Meta.csv"
thermistor_file = data_dir+"DA1_CRiver_TM_T_Final.csv"
sws1_file = data_dir+"SWS-1.csv"
sws1_joblib = data_dir+"SWS-1.joblib"
remove_index = 42  # hard wired
da1_joblib = data_dir+"da1.joblib"

with open(sws1_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = np.array(reader[1:])
data[data == "NA"] = np.nan
sws1 = dict()
sws1["time"] = np.array([datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                         for x in data[:, 0]])
sws1["level"] = data[:, 3].astype(float)
sws1["temperature"] = data[:, 1].astype(float)
joblib.dump(sws1, sws1_joblib)

# read coordinates
with open(coord_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = reader[1:]
da1 = dict()
for idata in data:
    therm_name = idata[1].replace("(", "").replace(")", "")
    da1[therm_name] = dict()
    da1[therm_name]["easting"] = float(idata[4])
    da1[therm_name]["northing"] = float(idata[5])
    da1[therm_name]["riverbed"] = float(idata[2])
    da1[therm_name]["depth"] = float(idata[3])*0.01
    da1[therm_name]["elevation"] = float(idata[2])-float(idata[3])*0.01

# read temperature data
# hard coded part,remove
with open(thermistor_file, 'r') as f:
    reader = list(csv.reader(f))
    header = [x.replace(".", "") for x in reader[0]]
    data = np.array(reader[1:])

# remove failed thermistor
header.pop(remove_index)
data = np.delete(data, remove_index, 1)

# clean up data, remove non-thermistor columns
header = header[4:]
time = np.array([datetime.strptime(x, "%m/%d/%Y %H:%M") for x in data[:, 0]])
data = data[:, 4:]
data[data == "NAN"] = np.nan
data[data == "NA"] = np.nan
data = data.astype(float)
data[data <= 0] = np.nan

# fill to regular data space
filled_time = np.array([time[0]+timedelta(seconds=x)
                        for x in
                        np.arange(0, (time[-1]-time[0]).total_seconds()+300, 300)])
ntime = len(filled_time)
nda1 = data.shape[1]
filled_data = np.empty((ntime, nda1))
filled_data[:] = np.nan
filled_data[filled_time.searchsorted(time)] = data

# fill in da1 dict
for ikey in list(da1.keys()):
    print(ikey)
    if ikey in header:
        da1[ikey]["temperature"] = filled_data[
            :, header.index(ikey)].astype(float)
    else:
        da1.pop(ikey)

# # # store time
da1["time"] = filled_time
joblib.dump(da1, da1_joblib)
