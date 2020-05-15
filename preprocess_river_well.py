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
import glob


def find_match(array_a, array_b):
    """
    find index of array_a, array_b,
    and return the indexes of (first) matched elements 
    array_a and array_b should be sorted 1d array
    """
    len_b = len(array_b)
    index_a = []
    index_b = []
    ind_b = 0
    for ind_a, value_a in enumerate(array_a):
        while (value_a >= array_b[ind_b]):
            if value_a == array_b[ind_b]:
                index_a.append(ind_a)
                index_b.append(ind_b)
            ind_b += 1
            if ind_b == len_b:
                return(np.array(index_a), np.array(index_b))


# directory
data_dir = "/mnt/e/dense_array/data/"
sws1_file = data_dir+"SWS-1.csv"
rg3_correction_file = data_dir+"RG3-T3/Data-correction/RG3_WL_Corrected.csv"
rg3_raw_file = data_dir+"RG3-T3/Data-processing/2020-02-21/RG3_Gage/RG3_Gage_Raw.csv"
air_file = data_dir+"300A_hsite_obs.csv"
well_dir = data_dir+"wells/"
wells = glob.glob(data_dir+"wells/399*csv")
wells = np.unique([iwell.split("/")[-1].split("_")[0] for iwell in wells])
river_joblib = data_dir+"river.joblib"
#wells = ["2-1", "2-2", "2-3"]


# read air_file
with open(air_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = np.array(reader[1:])
data[data == "NA"] = np.nan
air_time = np.array([datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S")
                     for x in data[:, 0]])
air_temperature = data[:, 5].astype(float)
filled_time = np.array(
    [air_time[0]+timedelta(seconds=x)
     for x in np.arange(0,
                        (air_time[-1]-air_time[0]).total_seconds()
                        + 900, 900)])
ntime = len(filled_time)
filled_time_index, air_time_index = find_match(filled_time, air_time)
filled_air_temperature = np.empty((ntime))
filled_air_temperature[:] = np.nan
filled_air_temperature[filled_time_index] = air_temperature[air_time_index]

air = dict()
air["time"] = filled_time
air["temperature"] = filled_air_temperature

joblib.dump(air, data_dir+"air.joblib")

# read sws1
with open(sws1_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = np.array(reader[1:])
data[data == "NA"] = np.nan
sws1_time = np.array([datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                      for x in data[:, 0]])
sws1_level = data[:, 3].astype(float)
sws1_temperature = data[:, 1].astype(float)
filled_time = np.array(
    [sws1_time[0]+timedelta(seconds=x)
     for x in np.arange(0,
                        (sws1_time[-1]-sws1_time[0]).total_seconds()
                        + 900, 900)])
ntime = len(filled_time)
filled_time_index, sws1_time_index = find_match(filled_time, sws1_time)
filled_sws1_level = np.empty((ntime))
filled_sws1_level[:] = np.nan
filled_sws1_level[filled_time_index] = sws1_level[sws1_time_index]
filled_sws1_temperature = np.empty((ntime))
filled_sws1_temperature[:] = np.nan
filled_sws1_temperature[filled_time_index] = sws1_temperature[sws1_time_index]


# read first part of rg3
with open(rg3_correction_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = np.array(reader[1:])
data[data == ""] = np.nan
corr_time = np.array([datetime.strptime(x, "%m/%d/%Y %H:%M")
                      for x in data[:, 0]])
corr_sws1 = data[:, -2].astype(float)
corr_rg3 = data[:, -3].astype(float)
filled_time_index, corr_time_index = find_match(filled_time, corr_time)
filled_river_level = np.empty((ntime))
filled_river_level[:] = np.nan
filled_river_level[filled_time_index] = corr_rg3[corr_time_index]
nonnan_index = ~np.isnan(corr_sws1*corr_rg3)
corr_sws1 = corr_sws1[nonnan_index]
corr_rg3 = corr_rg3[nonnan_index]

# read rg3 gauge
with open(rg3_raw_file, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = np.array(reader[1:])
data[data == "NA"] = np.nan
rg3_time = np.array([datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                     for x in data[:, 0]])
data = data[rg3_time <= corr_time[-1], :]
rg3_time = rg3_time[rg3_time <= corr_time[-1]]
rg3_temperature = data[:, -3].astype(float)
filled_time_index, rg3_time_index = find_match(filled_time, rg3_time)
filled_river_temperature = np.empty(ntime)
filled_river_temperature[:] = np.nan
filled_river_temperature[filled_time_index] = rg3_temperature[rg3_time_index]

# merge data
level_index = (np.isnan(filled_river_level))*(~np.isnan(filled_sws1_level))
filled_river_level[level_index] = filled_sws1_level[level_index] + \
    np.mean(corr_rg3-corr_sws1)
# np.interp( filled_sws1_level[level_index], corr_sws1, corr_rg3)
valid_temperature_index = (~np.isnan(filled_river_temperature)) * \
    (~np.isnan(filled_sws1_temperature))
rg3_valid_temperature = filled_river_temperature[valid_temperature_index]
sws1_valid_temperature = filled_sws1_temperature[valid_temperature_index]
temperature_index = (np.isnan(filled_river_temperature)) * \
    (~np.isnan(filled_sws1_temperature))
filled_river_temperature[temperature_index] = filled_sws1_temperature[temperature_index] + \
    np.mean(rg3_valid_temperature-sws1_valid_temperature)
# np.interp(filled_sws1_temperature[temperature_index], sws1_valid_temperature, rg3_valid_temperature)

river = dict()
river["time"] = filled_time
river["level"] = filled_river_level
river["temperature"] = filled_river_temperature
joblib.dump(river, river_joblib)

for iwell in wells:
    well_time = []
    well_level = []
    well_temperature = []
    well_spc = []
    for ifile in np.sort(glob.glob(well_dir + iwell+"_"+"*csv")):
        print(ifile)
        with open(ifile, 'r') as f:
            reader = list(csv.reader(f))
            header = reader[0]
            data = np.array(reader[1:])
        data[data == 'NA'] = np.nan
        well_time += [datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                      for x in data[:, 0]]
        well_temperature += data[:, 1].tolist()
        well_level += data[:, -1].tolist()
        well_spc += data[:, 2].tolist()

    well_time = np.array(well_time)
    well_temperature = np.array(well_temperature)
    well_level = np.array(well_level)
    well_spc = np.array(well_spc)
    filled_time = np.array(
        [well_time[0]+timedelta(seconds=x)
         for x in np.arange(0,
                            (well_time[-1]-well_time[0]).total_seconds()
                            + 900, 900)])
    ntime = len(filled_time)
    filled_time_index, well_time_index = find_match(filled_time, well_time)
    filled_well_level = np.empty((ntime))
    filled_well_level[:] = np.nan
    filled_well_level[filled_time_index] = well_level[well_time_index]
    filled_well_temperature = np.empty((ntime))
    filled_well_temperature[:] = np.nan
    filled_well_temperature[filled_time_index] = well_temperature[well_time_index]
    filled_well_spc = np.empty((ntime))
    filled_well_spc[:] = np.nan
    filled_well_spc[filled_time_index] = well_spc[well_time_index]

    well = dict()
    well["time"] = filled_time
    well["level"] = filled_well_level
    well["temperature"] = filled_well_temperature
    well["spc"] = filled_well_spc
    joblib.dump(well, data_dir+"wells/"+iwell+".joblib")
