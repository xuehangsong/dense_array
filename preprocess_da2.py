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


data_dir = "/mnt/e/dense_array/data/"
da2_file = data_dir+"Composite_Sampling_sheet_array(VGC).csv"

with open(da2_file, 'r') as f:
    reader = list(csv.reader(f))
    header = [x.replace(".", "") for x in reader[0]]
    data = np.array(reader[1:])
sample_start = [datetime.strptime(
    x+" "+y.split("-")[0],
    "%Y-%m-%d %H:%M") for x, y in zip(data[:, 0], data[:, 3])]
sample_end = [datetime.strptime(
    x+" "+y.split("-")[1],
    "%Y-%m-%d %H:%M") for x, y in zip(data[:, 0], data[:, 3])]
time = [x+0.5*(y-x) for x, y in zip(sample_start, sample_end)]
do = [float(x) for x in data[:, 32]]
do_satu = [float(x) for x in data[:, 31]]
spc = [float(x) for x in data[:, 33]]
spc_lab = [float(x) for x in data[:, 5]]
temp = [float(x) for x in data[:, 34]]
elevation = [float(x) if "River" not in x else np.nan
             for x in data[:, 28]]
name = [x for x in data[:, 27]]
da2 = dict()
for idata in np.arange(len(name)):
    if name[idata] not in da2.keys():
        da2[name[idata]] = dict()
        da2[name[idata]]["time"] = time[idata]
        da2[name[idata]]["do"] = do[idata]
        da2[name[idata]]["satu"] = do_satu[idata]
        da2[name[idata]]["spc"] = spc[idata]
        da2[name[idata]]["spc_lab"] = spc_lab[idata]
        da2[name[idata]]["temp"] = temp[idata]
        da2[name[idata]]["elevation"] = elevation[idata]
    else:
        da2[name[idata]]["time"] = np.append(da2[name[idata]]["time"],
                                             time[idata])
        da2[name[idata]]["do"] = np.append(da2[name[idata]]["do"],
                                           do[idata])
        da2[name[idata]]["satu"] = np.append(da2[name[idata]]["satu"],
                                             do_satu[idata])
        da2[name[idata]]["spc"] = np.append(da2[name[idata]]["spc"],
                                            spc[idata])
        da2[name[idata]]["spc_lab"] = np.append(da2[name[idata]]["spc_lab"],
                                                spc_lab[idata])
        da2[name[idata]]["temp"] = np.append(
            da2[name[idata]]["temp"], temp[idata])
        da2[name[idata]]["elevation"] = np.append(
            da2[name[idata]]["elevation"],
            elevation[idata])
