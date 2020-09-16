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

data_dir = "/mnt/e/dense_array/data/"
ms5_file = data_dir+"Last point of MS5 matched with solenoid.csv"
ms5_joblib = data_dir+"ms5.joblib"


with open(ms5_file, 'r') as f:
    reader = list(csv.reader(f))
    header = [x.replace(".", "") for x in reader[0]]
    data = np.array(reader[1:])
times = [datetime.strptime(
    x, "%Y-%m-%d %H:%M:%S") for x in data[:, 0]]
temp = [float(x) for x in data[:, 1]]
spc = [float(x) if x != "NA" else np.nan for x in data[:, 2]]
do_satu = [float(x) if x != "NA" else np.nan for x in data[:, 3]]
do = [float(x) if x != "NA" else np.nan for x in data[:, 4]]
name = [x for x in data[:, 6]]

ms5 = dict()
for idata in np.arange(len(name)):
    if name[idata] not in ms5.keys():
        ms5[name[idata]] = dict()
        ms5[name[idata]]["times"] = times[idata]
        ms5[name[idata]]["do"] = do[idata]
        ms5[name[idata]]["do_satu"] = do_satu[idata]
        ms5[name[idata]]["spc"] = spc[idata]
        ms5[name[idata]]["temp"] = temp[idata]
    else:
        ms5[name[idata]]["times"] = np.append(ms5[name[idata]]["times"],
                                              times[idata])
        ms5[name[idata]]["do"] = np.append(ms5[name[idata]]["do"],
                                           do[idata])
        ms5[name[idata]]["do_satu"] = np.append(ms5[name[idata]]["do_satu"],
                                                do_satu[idata])
        ms5[name[idata]]["spc"] = np.append(ms5[name[idata]]["spc"],
                                            spc[idata])
        ms5[name[idata]]["temp"] = np.append(
            ms5[name[idata]]["temp"], temp[idata])
joblib.dump(ms5, ms5_joblib)
