# SUMMARY:      preprocess_data.py
# USAGE:        main scripts to do the rtd analyze
# ORG:          Pacific Northwest National Laboratory
# AUTHOR:       Xuehang Song
# E-MAIL:       xuehang.song@pnnl.gov
# ORIG-DATE:    Mar-2020
# DESCRIPTION:
# DESCRIPTION-END

# from fastdtw import fastdtw

import pywt
import multiprocessing as mp
from matplotlib.patches import Rectangle
from pykrige.uk import UniversalKriging
from pykrige.uk3d import UniversalKriging3D
from scipy.interpolate import interp2d
from xml.dom import minidom
from xml.etree import ElementTree as ET
from scipy.interpolate import RectBivariateSpline
import h5py as h5
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from matplotlib.pyplot import cm
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import mpl_scatter_density
from sklearn.externals import joblib
import csv
import copy
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates


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


def plot_cwt_long(fig_name, wavelet_coef):
    """
    plot multiple year cwt
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wavelet_coef["time"],
            np.log10(wavelet_coef["arch"]),
            color="white",
            lw=2,
            linestyle="--")
    ax.fill_between(wavelet_coef["time"],
                    np.log10(wavelet_coef["arch"]),
                    len(wavelet_coef["time"]) *
                    np.log10(wavelet_coef["period"][-1]),
                    facecolor="white",
                    alpha=0.5,
                    zorder=10000)
    cf = ax.contourf(wavelet_coef["time"],
                     np.log10(wavelet_coef["period"]),
                     np.log10(wavelet_coef["power"]),
                     levels=np.linspace(-6, 0., 100),
                     extend="both",
                     cmap=plt.cm.jet,
                     zorder=1
                     )
    cb = plt.colorbar(cf, ax=ax, format="%.f", pad=0.01)
    cb.ax.set_ylabel("Wavelet power spectrum (log10)",
                     rotation=270, labelpad=20)
    cb.set_ticks(-np.arange(7)[::-1])
    ax.set_xlim(wavelet_coef["time"][0], wavelet_coef["time"][-1])
    ax.set_ylim(np.log10(wavelet_coef["period"][0]),
                np.log10(wavelet_coef["period"][-1]))
    ax.set_xlabel('Date')
    ax.set_yticks(np.log10([1/24, 1, 7, 30, 365]))
    ax.set_yticklabels(["1h", "1d", "1w", "1m", "1y"])
    ax.set_ylabel('Period (log10)')
    fig.set_size_inches(10, 3)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=600, bbox_inches=0)
    plt.close(fig)
    print("Hello World!")


def plot_cwt_short(fig_name, wavelet_coef, title):
    """
    plot multiple year cwt
    """

    date_label = ["2018-03-01",
                  "2018-06-01",
                  "2018-09-01",
                  "2018-12-01"]
    date_label_loc = [datetime.strptime(x, "%Y-%m-%d")
                      for x in date_label]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wavelet_coef["time"],
            np.log10(wavelet_coef["arch"]),
            color="white",
            lw=2,
            linestyle="--")
    ax.fill_between(wavelet_coef["time"],
                    np.log10(wavelet_coef["arch"]),
                    len(wavelet_coef["time"]) *
                    np.log10(wavelet_coef["period"][-1]),
                    facecolor="white",
                    alpha=0.5,
                    zorder=10000)
    cf = ax.contourf(wavelet_coef["time"],
                     np.log10(wavelet_coef["period"]),
                     np.log10(wavelet_coef["power"]),
                     levels=np.linspace(-6, 0., 100),
                     extend="both",
                     cmap=plt.cm.jet,
                     zorder=1
                     )
    cb = plt.colorbar(cf, ax=ax, format="%.f", pad=0.02)
    cb.ax.set_ylabel("Wavelet power spectrum (log10)",
                     rotation=270, labelpad=15)
    cb.set_ticks(-np.arange(7)[::-1])
    ax.set_xlim(wavelet_coef["time"][0], wavelet_coef["time"][-1])
    ax.set_ylim(np.log10(wavelet_coef["period"][0]),
                np.log10(wavelet_coef["period"][-1]))
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.set_xticks(date_label_loc)
    ax.set_xticklabels(date_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_yticks(np.log10([1/24, 1, 7, 30, 365]))
    ax.set_yticklabels(["1h", "1d", "1w", "1m", "1y"])
    ax.set_ylabel('Period (log10)')
    fig.set_size_inches(5.2, 3)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=600, bbox_inches=0)
    plt.close(fig)
    print("Hello World!")


def dense_array_cwt(wavelet_time, wavelet_data, nscale):
    """
    this function is based for analysis for dense array data (measuremnt frequency is 900 seconds)
    """

    # pywt.scale2frequency("morl",[0.40625,1.95e4])/24
    # (1/(pywt.scale2frequency("morl",[1.625,5.698e4])/900))/3600/24/365.25
    # the scale was chose to make periods start from 1800 seconds, end ends at 2 yr
    scale_lower = 1.625
    scale_upper = 5.698e4
    scale = np.logspace(np.log10(scale_lower),
                        np.log10(scale_upper),
                        nscale,
                        base=10)

    # for this study, the sample frequency is 900
    coef, freq = pywt.cwt(
        wavelet_data,  scale, "morl", 900)
    power = (np.abs(coef))**2/(scale[:, None])

    period = 1/freq/3600/24
    wavelet_delta_time = np.array([(x-wavelet_time[0]).total_seconds()
                                   for x in wavelet_time])/3600/24
    wavelet_arch = [np.min([
        period[-1],
        2*x,
        (wavelet_delta_time[-1]-x)*2])
        for x in wavelet_delta_time]

    # put data in pickle file
    cwt_coef = dict()
    cwt_coef["time"] = wavelet_time
    cwt_coef["delta_time"] = wavelet_delta_time
    cwt_coef["arch"] = wavelet_arch
    cwt_coef["freq"] = freq
    cwt_coef["scale"] = scale
    cwt_coef["coef"] = coef
    cwt_coef["power"] = power
    cwt_coef["period"] = period

    return(cwt_coef)


def prettify(element):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def DTWDistance(s1, s2):
    """
    The basic to caculate norm2 DTW.
    reference:"https://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/tree/master/"
    For large dataset, we can use the dtw package in python
    Xuehang Song
    07/19/2019
    """
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],
                                     DTW[(i, j-1)], DTW[(i-1, j-1)])
    return (DTW[len(s1)-1, len(s2)-1])**0.5


def dtw():
    # dtw (slow version)
    # ri = random.choices(np.arange(len(rive_rtracer[0])), k=50)
    # dist_spatial = [DTWDistance(river_tracer[:, ri[i]], river_tracer[:, ri[j]])
    #              for i in range(len(ri)) for j in range(len(ri))]
    ri = random.choices(np.arange(len(river_tracer[0])), k=400)
    dist_spatial_tri = [[fastdtw(river_tracer[:, ri[i]], river_tracer[:, ri[j]])[0]
                         for i in range(j+1)] for j in range(len(ri))]
    dist_spatial = np.zeros((len(ri), len(ri)))
    for i, j in enumerate(dist_spatial_tri):
        dist_spatial[i][0:len(j)] = j
    dist_spatial = dist_spatial+np.transpose(dist_spatial)
    dist_spatial = distance.squareform(
        dist_spatial, force="tovector", checks=True)


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
        elif delta_format == "secons":
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
    return(y)


# input data
data_dir = "/mnt/e/dense_array/data/"
geo_unit_file = data_dir+"300A_EV_surfaces_012612.dat"
bathymetry_file = data_dir+"g_bathymetry_v3_clip_2nd.asc"
da1_joblib = data_dir+"da1.joblib"
river_joblib = data_dir+"river.joblib"
air_joblib = data_dir+"air.joblib"
well2_1_joblib = data_dir+"well_2-1.joblib"
well2_2_joblib = data_dir+"well_2-2.joblib"
well2_3_joblib = data_dir+"well_2-3.joblib"
truncated_da1_joblib = data_dir+"truncated_da1.joblib"
well_file = "/mnt/e/rtd/Data/Observation_Data/well_coordinates_all.csv"
# output
results_dir = "/mnt/e/dense_array/results/"

# figure
img_dir = "/mnt/e/dense_array/figures/"

# output
paraview_dir = "/mnt/e/dense_array/paraview/"
material_h5 = paraview_dir+"material_ids.h5"
material_xdmf = paraview_dir+"material_ids.xdmf"
original_material_h5 = paraview_dir+"original_material_ids.h5"
original_materiall_xdmf = paraview_dir+"original_material_ids.xdmf"
da1_h5 = paraview_dir+"da1.h5"
da1_xdmf = paraview_dir+"da1.xdmf"
temperature_3d_h5 = paraview_dir+"3d_temperature.h5"
temperature_3d_xdmf = paraview_dir+"3d_temperature_"
mean_temperature_h5 = paraview_dir+"mean_temperature.h5"
mean_temperature_xdmf = paraview_dir+"mean_temperature.xdmf"


# load preprocessed data
# da1 = joblib.load(da1_joblib)
# river = joblib.load(river_joblib)


date_origin = datetime.strptime("2017-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
date_label = ["2017",
              "2018",
              "2019",
              "2020"]
date_label_loc = [datetime.strptime(x, "%Y") for x in date_label]

# date_label_loc = batch_time_to_delta(date_origin, date_label,
#                                      "%Y") / 3600


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
material_dz = np.array([0.1]*material_nz)
bath_x = np.arange(593000.5598+0.5, 594900.5598, 1)
bath_y = np.arange(114500.6771+0.5, 117800.6771, 1)


def plot_depth_sorted():
    """
    plot temperature sorted by thermistor depth
    """
    da1 = joblib.load(da1_joblib)
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    therm_depth = [da1[x]["depth"] for x in thermistors]
    thermistors = [thermistors[x] for x in np.argsort(therm_depth)]
    therm_depth = [therm_depth[x] for x in np.argsort(therm_depth)]

    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_low = np.nanmin(therm_data, 0)
    therm_high = np.nanmax(therm_data, 0)

    ncol = 10
    nrow = np.ceil(ntherm/ncol).astype(int)
    fig_name = img_dir + "da1_depth_sorted.png"
    fig, axes = plt.subplots(nrow, ncol)
    for therm_index, ithermistor in enumerate(thermistors):
        ax = axes.flatten()[therm_index]
        ax.fill_between(da1["time"], therm_low, therm_high, color="lightgrey")
        ax.plot(
            da1["time"], da1[ithermistor]["temperature"], color="black")
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature $_oC$')
        ax.set_xticks(date_label_loc)
        ax.set_xticklabels(date_label)
        ax.set_ylim(0, 25)
        ax.set_title(
            "Depth ="+"{0:.3f}".format(da1[ithermistor]["depth"])+" m")
        # ax.set_xlim(min(time_label_loc), max(time_label_loc))
        # ax.legend(frameon=False, loc="upper left")
    for ax in axes.flatten()[therm_index+1:ncol*nrow]:
        ax.set_axis_off()
    fig.set_size_inches(30, 20)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches=0)  # , transparent=True)
    plt.close(fig)
    print("Hello World!")


def plot_depth_sorted_fill():
    """
    plot temperature sorted by thermistor depth
    """

    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    therm_depth = [da1[x]["depth"] for x in thermistors]
    thermistors = [thermistors[x] for x in np.argsort(therm_depth)]
    therm_depth = [therm_depth[x] for x in np.argsort(therm_depth)]

    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_low = np.nanmin(therm_data, 0)
    therm_high = np.nanmax(therm_data, 0)

    ncol = 10
    nrow = np.ceil(ntherm/ncol).astype(int)
    fig_name = img_dir + "da1_depth_sorted_back.png"
    fig, axes = plt.subplots(nrow, ncol)
    for therm_index, ithermistor in enumerate(thermistors[0:2]):
        ax = axes.flatten()[therm_index]
        ax.fill_between(da1["time"], therm_low, therm_high, color="lightgrey")
        ax.plot(
            da1["time"], da1[ithermistor]["temperature"], color="blue")
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature $_oC$')
        ax.set_xticks(date_label_loc)
        ax.set_xticklabels(date_label)
        ax.set_ylim(0, 25)
        ax.set_title(
            "Depth ="+"{0:.3f}".format(da1[ithermistor]["depth"])+" m")
        # ax.set_xlim(min(time_label_loc), max(time_label_loc))
        # ax.legend(frameon=False, loc="upper left")
    for ax in axes.flatten()[therm_index+1:ncol*nrow]:
        ax.set_axis_off()
    fig.set_size_inches(30, 20)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches=0)  # , transparent=True)
    plt.close(fig)
    print("Hello World!")


def plot_mean_variance_backup():
    """
    plot elevation sorted by thermistor depth
    """

#    date_end = datetime.strptime("2019-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    thermistors = [thermistors[x] for x in np.argsort(therm_elevation)]
    therm_elevation = [da1[x]["elevation"] for x in thermistors]
    therm_elevation = [therm_elevation[x] for x in np.argsort(therm_elevation)]
    therm_depth = [da1[x]["depth"] for x in thermistors]
    therm_riverbed = [da1[x]["riverbed"] for x in thermistors]

    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_time = da1["time"]
#    therm_data = therm_data[:, da1["time"] < date_end]
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm == ntherm)[0]

    mean_temp = np.mean(therm_data[:, valid_index], 1)
    std_temp = np.std(therm_data[:, valid_index], 1)
    cv_temp = std_temp/mean_temp

    imgfile = img_dir+"mean_temp_vs_depth.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(therm_depth, cv_temp)
    fig.set_size_inches(4, 4)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"mean_temp_vs_depth.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(mean_temp, std_temp)
    ax.set_aspect(1)
    fig.set_size_inches(4, 4)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)


def plot_elevation_sorted():
    """
    plot elevation sorted by thermistor depth
    """
    da1 = joblib.load(da1_joblib)

    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    therm_elevation = [da1[x]["elevation"] for x in thermistors]
    thermistors = [thermistors[x] for x in np.argsort(therm_elevation)]
    therm_elevation = [therm_elevation[x] for x in np.argsort(therm_elevation)]

    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_low = np.nanmin(therm_data, 0)
    therm_high = np.nanmax(therm_data, 0)

    ncol = 10
    nrow = np.ceil(ntherm/ncol).astype(int)
    fig_name = img_dir + "da1_elevation_sorted.png"
    fig, axes = plt.subplots(nrow, ncol)
    for therm_index, ithermistor in enumerate(thermistors):
        ax = axes.flatten()[therm_index]
        ax.fill_between(da1["time"], therm_low, therm_high, color="lightgrey")
        ax.plot(
            da1["time"], da1[ithermistor]["temperature"], color="black")

        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature $_oC$')
        ax.set_xticks(date_label_loc)
        ax.set_xticklabels(date_label)
        ax.set_ylim(0, 25)
        ax.set_title(
            "Elevation ="+"{0:.3f}".format(da1[ithermistor]["elevation"])+" m")
        # ax.set_xlim(min(time_label_loc), max(time_label_loc))
        # ax.legend(frameon=False, loc="upper left")
    for ax in axes.flatten()[therm_index+1:ncol*nrow]:
        ax.set_axis_off()
    fig.set_size_inches(30, 20)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches=0)  # , transparent=True)
    plt.close(fig)
    print("Hello World!")


def create_material():
    """
    read in data file and create material file
    """

    # load unit data,create material
    unit_data = np.genfromtxt(geo_unit_file, skip_header=21)
    unit_data = unit_data[(unit_data[:, 0] >= west-20)
                          * (unit_data[:, 0] <= east+20), :]
    unit_data = unit_data[(unit_data[:, 1] >= south-20)
                          * (unit_data[:, 1] <= north+20), :]
    unit_x = np.sort(np.unique(unit_data[:, 0]))
    unit_y = np.sort(np.unique(unit_data[:, 1]))
    unit_ringold = unit_data[
        :, -1].reshape((len(unit_x), len(unit_y)), order="F")
    interp_ringold = RectBivariateSpline(
        unit_x, unit_y, unit_ringold, kx=1, ky=1)
    ringold_ele = interp_ringold(x, y)
    ringold_array = np.dstack([ringold_ele]*material_nz)

    # load river bathmetry
    bath_data = np.transpose(np.genfromtxt(bathymetry_file, skip_header=6))
    bath_data = bath_data[:, ::-1]
    interp_bath = RectBivariateSpline(bath_x, bath_y, bath_data, kx=1, ky=1)
    riverbed_ele = interp_bath(x, y)
    riverbed_ele[riverbed_ele < 0] = np.nanmax(riverbed_ele)
    riverbed_array = np.dstack([riverbed_ele]*material_nz)

    # plot bathmetry map
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    therm_x = np.array([da1[x]["easting"] for x in thermistors])
    therm_y = np.array([da1[x]["northing"] for x in thermistors])
    imgfile = img_dir+"original_bathymetry.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(therm_x,
               therm_y,
               color="black",
               label="Themistors",
               zorder=1000,
               s=15)
    ax.legend(loc="upper right")  # , zorder=20)
    cf = ax.contourf(
        x,
        y,
        np.transpose(riverbed_ele),
        levels=np.arange(103, 108.05, 0.05),
        extend="both",
        zorder=0,
        cmap=plt.cm.terrain)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_xticks(np.arange(594450, 596450, 10))
    ax.set_xlim(ox, ex)
    ax.set_ylim(oy, ey)
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.set_aspect(1)
    ax4 = fig.add_axes([0.80, 0.118, 0.03, 0.83])
    cb = plt.colorbar(cf, cax=ax4, extend="both")
    cb.ax.set_ylabel("Bathymetry (m)",
                     rotation=270, labelpad=14)
    cb.set_ticks(np.arange(103, 108.5, 0.5))
    fig.set_size_inches(8, 10)
    fig.subplots_adjust(left=0.12,
                        right=0.8,
                        bottom=0.12,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)

    # #load thermistor location
    # thermistors = [x for x in list(da1.keys()) if x != "time"]
    # ntherm = len(thermistors)
    # therm_riverbed = np.array([da1[x]["riverbed"] for x in thermistors])
    # therm_x = np.array([da1[x]["easting"] for x in thermistors])
    # therm_y = np.array([da1[x]["northing"] for x in thermistors])

    # # use RectBivariateSpline instead interp2d as RectBivariateSpline is much fastdtw
    # therm_x_sorted = np.sort(therm_x).tolist()
    # therm_y_sorted = np.sort(therm_y).tolist()
    # therm_interp_riverbed_all = interp_bath(therm_x_sorted, therm_y_sorted)
    # therm_interp_riverbed = np.array([therm_interp_riverbed_all[therm_x_sorted.index(
    #     therm_x[i]), therm_y_sorted.index(therm_y[i])] for i in range(ntherm)])
    # therm_riverbed_bias = therm_interp_riverbed-therm_riverbed
    # therm_riverbed_bias = therm_riverbed_bias[np.abs(therm_riverbed_bias) < 20]

    # # find locations near the thermistor (distance<4 m)
    # bath_x_flatten = np.array(bath_x.tolist()*len(bath_y))
    # bath_y_flatten = np.array([[x]*len(bath_x)
    #                            for x in bath_y]).flatten(order="C")
    # bath_data_flatten = bath_data.flatten(order="F")
    # cells_to_update = [np.where(((bath_x_flatten-therm_x[i])**2 +
    #                              (bath_y_flatten-therm_y[i])**2) < 16)[0].tolist()
    #                    for i in range(ntherm)]
    # cells_to_update = np.unique([x for y in cells_to_update for x in y])
    # output_x = bath_x_flatten[cells_to_update]
    # output_y = bath_y_flatten[cells_to_update]
    # output_x_unique = np.unique(output_x).tolist()
    # output_y_unique = np.unique(output_y).tolist()

    # # find location around the to-be-updated cells
    # cells_for_interp = [np.where(
    #     (((bath_x_flatten-therm_x[i])**2 +
    #       (bath_y_flatten-therm_y[i])**2) >= 16) *
    #     (((bath_x_flatten-therm_x[i])**2 +
    #       (bath_y_flatten-therm_y[i])**2) <= 1024))[0].tolist() for i in range(ntherm)]
    # cells_for_interp = np.unique([x for y in cells_for_interp for x in y])
    # # create interplation function
    # input_x = np.append(bath_x_flatten[cells_for_interp], therm_x)
    # input_y = np.append(bath_y_flatten[cells_for_interp], therm_y)
    # input_z = np.append(bath_data_flatten[cells_for_interp], therm_riverbed)
    # input_x = input_x[input_z > 0]
    # input_y = input_y[input_z > 0]
    # input_z = input_z[input_z > 0]
    # interp2d_bath = interp2d(input_x, input_y, input_z,
    #                          fill_value=np.nan, kind="linear")
    # # update
    # bath_update_all = np.transpose(
    #     interp2d_bath(output_x_unique, output_y_unique))
    # bath_update = np.array([bath_update_all[
    #     output_x_unique.index(x),
    #     output_y_unique.index(y)]
    #     for x, y in zip(output_x, output_y)])

    # create material
    z_array = np.array([material_z]*nx*ny).reshape((nx, ny, material_nz))
    material_ids = np.zeros((nx, ny, material_nz))
    material_ids[z_array <= riverbed_array] = 1
    material_ids[z_array <= ringold_array] = 4
    material_ids = material_ids.astype("int")

    # reshape material to fit xdmf format
    material_ids = material_ids.flatten(
        order="F")
    # ouput material file
    hdf5 = h5.File(original_material_h5, "w")
    hdf5.create_dataset("Materials", data=material_ids)
    hdf5.close()

    # output xdmf
    xml_root = ET.Element("Xdmf", Version="3.0")
    xml_domain = ET.SubElement(xml_root, "Domain")

    xml_grid = ET.SubElement(xml_domain, "Grid",
                             {'Name': "material",
                              'GridType': 'Uniform'})

    xml_toplogoy = ET.SubElement(xml_grid, "Topology",
                                 {'TopologyType': '3DRECTMesh',
                                  'Dimensions': "{0} {1} {2}".format(material_nz, ny, nx)})
    xml_geometry = ET.SubElement(xml_grid, 'Geometry',
                                 {'GeometryType': "VXVYVZ"})
    xml_geometry_x = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(nx),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_x.text = np.array_str(x).strip("[]").replace("\n", " ")
    xml_geometry_y = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(ny),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_y.text = np.array_str(y).strip("[]").replace("\n", " ")
    xml_geometry_z = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(material_nz),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_z.text = np.array_str(
        material_z).strip("[]").replace("\n", " ")
    xml_material = ET.SubElement(xml_grid, "Attribute",
                                 {"Name": "Materials",
                                  "AttributeType": "Scalar",
                                  "Center": "Node"})
    material_dataitem = ET.SubElement(xml_material, "DataItem",
                                      {"Format": "HDF",
                                       "NumberType": "Float",
                                       "Precision": "8",
                                       "Dimensions": "{0} {1} {2}".format(nx, ny, material_nz)})
    material_dataitem.text = original_material_h5.split(
        "/")[-1]+":/Materials"
    with open(original_materiall_xdmf, 'w') as f:
        f.write(prettify(xml_root))


def clustering():

    # clustering the termistor based on their correlation
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    date_end = datetime.strptime("2019-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    ntherm = len(thermistors)
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_data = therm_data[:, da1["time"] < date_end]
    therm_depth = [da1[x]["depth"] for x in thermistors]
    therm_elevation = [da1[x]["elevation"] for x in thermistors]
    therm_riverbed = [da1[x]["riverbed"] for x in thermistors]
    # # choose time segments (>30 days) with all thermistors functional
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm == ntherm)[0]
    valid_seg = np.where(np.diff(valid_index) > 1)[0]
    segment_start = valid_index[valid_seg+1]
    segment_end = valid_index[np.append(valid_seg[1:], -1)]
    segments = [np.arange(start, end)
                for start, end in zip(segment_start, segment_end)]
    segment_length = [len(x)*5/60/24 for x in segments]
    segments = [x for x, y in zip(segments, segment_length) if y >= 0]
    clustering_index = np.sort([x for y in [x.tolist()
                                            for x in segments] for x in y])

    dist_cor_da1 = distance.pdist(
        therm_data[:, clustering_index], metric='correlation')

    cluster_cor_da1 = hierarchy.ward(dist_cor_da1)

    colors = cm.Set1(np.linspace(0, 1, 9))[
        np.array([6, 2, 1, 3, 4, 0, 7, 8, 5])]

    colors = np.array(["red", "orange", "green", "blue", "black"])

    hierarchy.set_link_color_palette([rgb2hex(x)
                                      for x in colors])
    ncluster = 5

    cluster_id_da1 = hierarchy.fcluster(
        cluster_cor_da1, t=ncluster, criterion="maxclust")

    # plot dendrogram map
    imgfile = img_dir+"dendrogram.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    dendrogram(cluster_cor_da1,
               truncate_mode='level',
               count_sort=False,
               p=9,
               color_threshold=0.12,
               labels=thermistors,
               above_threshold_color='grey',
               orientation="top")
    legend_elements = list()
    for icluster in range(ncluster):
        legend_elements.append(
            Line2D([0], [0],
                   color=colors[icluster],
                   label='Spatial cluster #'+str(icluster+1)))
    legend1 = plt.legend(handles=legend_elements,
                         fontsize=8,
                         loc='upper right')
    ax.add_artist(legend1)
    ax.set_xlabel("Index of thermistor (-)")
    ax.set_ylabel("Height")
    fig.set_size_inches(8, 4)
    plt.subplots_adjust(top=0.98, bottom=0.25, left=0.1,
                        right=0.98, wspace=0.0, hspace=0.05)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    # plot 3d cluster map
    imgfile = img_dir+"spatial_cluster_da1_3d.png"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for ithermistor, icluster in enumerate(cluster_id_da1):
        ax.scatter(da1[thermistors[ithermistor]]["easting"],
                   da1[thermistors[ithermistor]]["northing"],
                   da1[thermistors[ithermistor]]["elevation"],
                   color=colors[icluster-1],
                   # edgecolor=colors[icluster-1],
                   # facecolor="none",
                   s=100)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.set_size_inches(6.7, 9)
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)

    # plot 2d cluster map
    imgfile = img_dir+"spatial_cluster_da1_2d.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ithermistor, icluster in enumerate(cluster_id_da1):
        ax.scatter(da1[thermistors[ithermistor]]["easting"],
                   da1[thermistors[ithermistor]]["northing"],
                   edgecolor=colors[icluster-1],
                   facecolor="none",
                   lw=2,
                   s=100)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect(1)
    fig.set_size_inches(6.7, 9)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)

    # plot 2d cluster map
    imgfile = img_dir+"elevation_depth_cluster.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(therm_elevation, therm_depth,
               c=colors[np.array(cluster_id_da1-1)])
    # ax.set_xlabel("Easting (m)")
    # ax.set_ylabel("Northing (m)")
    # ax.set_aspect(1)
    fig.set_size_inches(6.7, 9)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)

    imgfile = img_dir+"riverbed_depth_cluster.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(therm_riverbed, therm_depth,
               c=colors[np.array(cluster_id_da1-1)])
    # ax.set_xlabel("Easting (m)")
    # ax.set_ylabel("Northing (m)")
    # ax.set_aspect(1)
    fig.set_size_inches(6.7, 9)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)


def clustering_paraview():

    # clustering the termistor based on their corelation
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_data = therm_data[:, ~np.isnan(np.sum(therm_data, 0))]
    dist_cor_da1 = distance.pdist(therm_data, metric='correlation')
    cluster_cor_da1 = hierarchy.ward(dist_cor_da1)

    cluster_id_da1 = []
    for ncluster in range(11)[2:]:
        cluster_id_da1 += [hierarchy.fcluster(
            cluster_cor_da1, t=ncluster, criterion="maxclust")]

    # dump cluster to hdf5
    dataset_coord = np.array([[da1[x]["easting"], da1[x]["northing"],
                               da1[x]["elevation"]] for x in thermistors])
    dataset_depth = np.array([da1[x]["depth"] for x in thermistors])
    dataset_cluster_id = np.array(cluster_id_da1)
    hdf5 = h5.File(da1_h5, "w")
    hdf5.create_dataset("Coordinates", data=dataset_coord)
    hdf5.create_dataset("Depths", data=dataset_depth)
    for ncluster, cluster_data in enumerate(cluster_id_da1):
        hdf5.create_dataset(str(ncluster+2)+"_clusters", data=cluster_data)
    hdf5.close()

    xml_root = ET.Element("Xdmf", Version="3.0")
    xml_domain = ET.SubElement(xml_root, "Domain")
    grid = ET.SubElement(xml_domain, "Grid",
                         {'Name': "Themistors",
                          'GridType': 'Uniform'})
    toplogoy = ET.SubElement(grid, "Topology",
                             {'TopologyType': 'Polyvertex',
                              'NodesPerElement': str(ntherm)})
    geometry = ET.SubElement(grid, 'Geometry',
                             {'GeometryType': "XYZ"})
    geometry_dataitem = ET.SubElement(geometry, 'DataItem',
                                      {'Dimensions': "{0} {1}".format(ntherm, 3),
                                       "NumberType": "Float",
                                       "Precision": "4",
                                       "Format": "HDF"})
    geometry_dataitem.text = da1_h5.split("/")[-1]+":/Coordinates"

    # depth as attributes
    depth = ET.SubElement(grid, "Attribute",
                          {"Name": "Depths",
                           "AttributeType": "Scalar",
                           "Center": "Node"})
    depth_dataitem = ET.SubElement(depth, 'DataItem',
                                   {'Dimensions': "{0} {1}".format(ntherm, 1),
                                    "NumberType": "Int",
                                    "Format": "HDF"})
    depth_dataitem.text = da1_h5.split("/")[-1]+":/Depths"

    # cluster_ids as attributes
    cluster_attr = []
    for iattr in np.arange(len(cluster_id_da1)):
        ncluster = iattr+2
        cluster_attr.append(ET.SubElement(grid, "Attribute",
                                          {"Name": str(ncluster)+"_clusters",
                                           "AttributeType": "Scalar",
                                           "Center": "Node"}))
        cluster_id_dataitem = ET.SubElement(cluster_attr[iattr], 'DataItem',
                                            {'Dimensions': "{0} {1}".format(ntherm, 1),
                                             "NumberType": "Int",
                                             "Format": "HDF"})
        cluster_id_dataitem.text = da1_h5.split(
            "/")[-1]+":/"+str(ncluster)+"_clusters"

    # ouput xmdf
    with open(da1_xdmf, 'w') as f:
        f.write(prettify(xml_root))


def create_material_updated():
    """
    read in data file and create material file
    """

    # load unit data,create material
    unit_data = np.genfromtxt(geo_unit_file, skip_header=21)
    unit_data = unit_data[(unit_data[:, 0] >= west-20)
                          * (unit_data[:, 0] <= east+20), :]
    unit_data = unit_data[(unit_data[:, 1] >= south-20)
                          * (unit_data[:, 1] <= north+20), :]
    unit_x = np.sort(np.unique(unit_data[:, 0]))
    unit_y = np.sort(np.unique(unit_data[:, 1]))
    unit_ringold = unit_data[
        :, -1].reshape((len(unit_x), len(unit_y)), order="F")
    interp_ringold = RectBivariateSpline(
        unit_x, unit_y, unit_ringold, kx=1, ky=1)
    ringold_ele = interp_ringold(x, y)
    ringold_array = np.dstack([ringold_ele]*material_nz)

    # load river bathmetry
    bath_data = np.transpose(np.genfromtxt(bathymetry_file, skip_header=6))
    bath_data = bath_data[:, ::-1]

    # load thermistor location
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    ntherm = len(thermistors)
    therm_riverbed = np.array([da1[x]["riverbed"] for x in thermistors])
    therm_x = np.array([da1[x]["easting"] for x in thermistors])
    therm_y = np.array([da1[x]["northing"] for x in thermistors])

    # find locations near the thermistor (distance<4 m) in the 1m bathmetry data
    bath_x_flatten = np.array(bath_x.tolist()*len(bath_y))
    bath_y_flatten = np.array([[x]*len(bath_x)
                               for x in bath_y]).flatten(order="C")
    bath_data_flatten = bath_data.flatten(order="F")
    bath_to_update = [np.where(((bath_x_flatten-therm_x[i])**2 +
                                (bath_y_flatten-therm_y[i])**2) < 25)[0].tolist()
                      for i in range(ntherm)]
    bath_to_update = np.unique([x for y in bath_to_update for x in y])

    # find locations near the thermistor (distance<4 m) in the model domain
    x_flatten = np.array(x.tolist()*ny)
    y_flatten = np.array([[s]*nx
                          for s in y]).flatten(order="C")
    cells_to_update = [np.where(((x_flatten-therm_x[i])**2 +
                                 (y_flatten-therm_y[i])**2) < 25)[0].tolist()
                       for i in range(ntherm)]
    cells_to_update = np.unique([x for y in cells_to_update for x in y])

    # check which bathmetry to update
    imgfile = img_dir+"updated_elevation.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(x_flatten,
               y_flatten,
               c="gold",
               s=2,
               label="1m bathemetry")
    ax.scatter(x_flatten[cells_to_update],
               y_flatten[cells_to_update],
               c="green",
               s=2,
               label="Krigged bathemetry")
    ax.scatter(therm_x,
               therm_y,
               c="black",
               label="Themistors",
               s=5)
    ax.add_patch(Rectangle(
        (ox, oy), ex-ox, ey-oy,
        fill=None,
        label="Dense array area",
        color="red", lw=2))
    ax.set_xlim(ox-20, ex+20)
    ax.set_ylim(oy-20, ey+20)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect(1)
    ax.legend(loc="upper right")
    scatter_index = np.logical_or(
        np.logical_or(
            np.logical_or(
                (bath_x_flatten < ox),
                (bath_x_flatten > ex)),
            (bath_y_flatten >= ey)),
        (bath_y_flatten < oy))
    ax.scatter(bath_x_flatten[scatter_index],
               bath_y_flatten[scatter_index],
               c="gold",
               s=2,
               label="1m bathemetry")
    fig.set_size_inches(7.5, 9)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    # kriging riverbed based on thermistor surveyed data
    UK_bath = UniversalKriging(
        therm_x,
        therm_y,
        therm_riverbed,
        variogram_model='linear',
        drift_terms=['regional_linear'])

    # generate new 1m bathemetry dataset
    updated_bath, ss = UK_bath.execute(
        "points",
        bath_x_flatten[bath_to_update],
        bath_y_flatten[bath_to_update])
    bath_data_flatten_new = copy.deepcopy(bath_data_flatten)
    bath_data_flatten_new[bath_to_update] = updated_bath.data
    bath_data_new = bath_data_flatten_new.reshape(
        (len(bath_x), len(bath_y)), order="F")

    # intepolate riverbed for model domain
    interp_bath = RectBivariateSpline(
        bath_x, bath_y, bath_data_new, kx=1, ky=1)
    riverbed_ele = interp_bath(x, y)
    riverbed_flatten = riverbed_ele.flatten(order="F")

    # update riverbed for model domain
    riverbed_update, ss = UK_bath.execute(
        "points",
        x_flatten[cells_to_update],
        y_flatten[cells_to_update])
    riverbed_flatten[cells_to_update] = riverbed_update.data
    riverbed_ele = riverbed_flatten.reshape((nx, ny), order="F")
    riverbed_ele[riverbed_ele < 0] = 105.78418  # hard coded
    riverbed_ele[riverbed_ele > 105.78418] = 105.78418

    riverbed_array = np.dstack([riverbed_ele]*material_nz)

    # plot bathmetry map
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    imgfile = img_dir+"bathymetry.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(therm_x,
               therm_y,
               color="black",
               label="Themistors",
               zorder=1000,
               s=15)
    ax.legend(loc="upper right")  # , zorder=20)
    cf = ax.contourf(
        x,
        y,
        np.transpose(riverbed_ele),
        levels=np.arange(103, 108.05, 0.05),
        extend="both",
        zorder=0,
        cmap=plt.cm.terrain)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_xticks(np.arange(594450, 596450, 10))
    ax.set_xlim(ox, ex)
    ax.set_ylim(oy, ey)
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.set_aspect(1)
    ax4 = fig.add_axes([0.80, 0.118, 0.03, 0.83])
    cb = plt.colorbar(cf, cax=ax4, extend="both")
    cb.ax.set_ylabel("Bathymetry (m)",
                     rotation=270, labelpad=14)
    cb.set_ticks(np.arange(103, 108.5, 0.5))
    fig.set_size_inches(8, 10)
    fig.subplots_adjust(left=0.12,
                        right=0.8,
                        bottom=0.12,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0,
                dpi=300)  # , transparent=True)
    plt.close(fig)

    # create material
    z_array = np.array([material_z]*nx*ny).reshape((nx, ny, material_nz))
    material_ids = np.zeros((nx, ny, material_nz))
    material_ids[z_array <= riverbed_array] = 1
    material_ids[z_array <= ringold_array] = 4
    material_ids = material_ids.astype("int")

    # reshape material to fit xdmf format
    material_ids = material_ids.flatten(
        order="F")
    # ouput material file
    hdf5 = h5.File(material_h5, "w")
    hdf5.create_dataset("Materials", data=material_ids)
    hdf5.close()

    # output xdmf
    xml_root = ET.Element("Xdmf", Version="3.0")
    xml_domain = ET.SubElement(xml_root, "Domain")

    xml_grid = ET.SubElement(xml_domain, "Grid",
                             {'Name': "material",
                              'GridType': 'Uniform'})

    xml_toplogoy = ET.SubElement(xml_grid, "Topology",
                                 {'TopologyType': '3DRECTMesh',
                                  'Dimensions': "{0} {1} {2}".format(material_nz, ny, nx)})
    xml_geometry = ET.SubElement(xml_grid, 'Geometry',
                                 {'GeometryType': "VXVYVZ"})
    xml_geometry_x = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(nx),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_x.text = np.array_str(x).strip("[]").replace("\n", " ")
    xml_geometry_y = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(ny),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_y.text = np.array_str(y).strip("[]").replace("\n", " ")
    xml_geometry_z = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(material_nz),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_z.text = np.array_str(
        material_z).strip("[]").replace("\n", " ")
    xml_material = ET.SubElement(xml_grid, "Attribute",
                                 {"Name": "Materials",
                                  "AttributeType": "Scalar",
                                  "Center": "Node"})
    material_dataitem = ET.SubElement(xml_material, "DataItem",
                                      {"Format": "HDF",
                                       "NumberType": "Float",
                                       "Precision": "8",
                                       "Dimensions": "{0} {1} {2}".format(nx, ny, material_nz)})
    material_dataitem.text = material_h5.split("/")[-1]+":/Materials"
    fname = material_xdmf
    with open(fname, 'w') as f:
        f.write(prettify(xml_root))


def kriging_temperature_only_thermistor():
    """
    krig temperature to the field.
    """

    da1 = joblib.load(da1_joblib)
    river = joblib.load(river_joblib)
    da1["delta_day"] = np.array(
        [(x-date_origin).total_seconds()/3600/24 for x in da1["time"]])

    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]
    ntherm = len(thermistors)
    therm_x = np.array([da1[x]["easting"] for x in thermistors])
    therm_y = np.array([da1[x]["northing"] for x in thermistors])
    therm_z = np.array([da1[x]["elevation"] for x in thermistors])
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])

    # choose time segments (>30 days) with all thermistors functional
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm > 64)[0]
    valid_seg = np.where(np.diff(valid_index) > 1)[0]
    segment_start = valid_index[valid_seg+1]
    segment_end = valid_index[np.append(valid_seg[1:], -1)]
    segments = [np.arange(start, end)
                for start, end in zip(segment_start, segment_end)]
    segment_length = [len(x)*5/60/24 for x in segments]
    segments = [x for x, y in zip(segments, segment_length) if y > 30]
#    print([len(x)*5/60/24 for x in segments])

    segments = [segments[0][0:2], segments[1][0:2]]

    # read material file
    hdf5 = h5.File(material_h5, "r")
    material_ids = hdf5["Materials"][:].reshape(
        (nx, ny, material_nz), order="F")
    hdf5.close()

    # use thermistor indicaoptr to cut part of the thermistor rods
    material_ids = material_ids[:, :, -nz:]
    temperature_indicator = np.zeros(material_ids.shape)
    temperature_indicator[material_ids > 0] = 1
    thickest_z = np.max(np.sum(temperature_indicator, 2))
    x_index, y_index = np.where(np.sum(temperature_indicator, 2) == thickest_z)
    intersect_x = max(x_index)
    intersect_y = max(y_index)
    # x1, y1 = x[intersect_x], y[0]
    # x2, y2 = x[0], y[intersect_y]
    x1, y1 = 594474, y[0]
    x2, y2 = x[0], 116304
    west_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) < 0]
    y2 += 18*(y2-y1)/(x1-x2)
    x1 += 18
    east_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) > 0]
    blank_index = np.array(east_blank+west_blank)
    temperature_indicator[blank_index[:, 0], blank_index[:, 1], :] = 0

    # thickness of current location
    thickness = np.array([
        [(np.cumsum(dz*(material_ids[ix, iy, ::-1]))[::-1]-0.5*dz).tolist()
         for iy in range(ny)] for ix in range(nx)])
    # temperature_indicator = np.ones(nx*ny*nz)
    # # find locations near the thermistor (distance<4 m) in the model domain
    # xyz_flatten = np.array([[ix, iy, iz] for iz in z for iy in y for ix in x])
    # cells_near_therm = [np.where(((xyz_flatten[:, 0]-therm_x[i])**2 +
    #                               (xyz_flatten[:, 1]-therm_y[i])**2 +
    #                               (xyz_flatten[:, 2]-therm_z[i])**2*25) < 25)[0].tolist()
    #                     for i in range(ntherm)]
    # cells_near_therm = np.unique([x for y in cells_near_therm for x in y])
    # temperature_indicator[cells_near_therm] = 1
    # temperature_indicator = temperature_indicator.reshape(
    #     (nx, ny, nz), order="C")
    # hdf5.create_dataset("Temperature_indicator", data=temperature_indicator)
    hdf5 = h5.File(temperature_3d_h5, "w")
    hdf5.create_dataset("Materials", data=material_ids.flatten(order="F"))
    hdf5.create_dataset("Temperature_indicator",
                        data=temperature_indicator.flatten(order="F"))
    hdf5.create_dataset("Thickness",
                        data=thickness.flatten(order="F"))

    for iseg in segments:
        print(len(iseg))
        for t_index in iseg:
            print(t_index)
            # krig data for one segments
            uk3d = UniversalKriging3D(
                therm_x,
                therm_y,
                therm_z,
                therm_data[:, t_index],
                variogram_model='linear',
                drift_terms=['regional_linear'])
            k3d, ss3d = uk3d.execute('grid', x, y, z)
            # output data to hdf5
            temp_data = k3d.data.swapaxes(0, 2).flatten(order="F")
            group = hdf5.create_group(str(round(da1["delta_day"][t_index], 3)))
            group.create_dataset("Temperature", data=temp_data)
    hdf5.close()

    # write xdmf files
    for seg_index, iseg in enumerate(segments):
        print(len(iseg))

        xml_root = ET.Element("Xdmf", Version="3.0")
        xml_domain = ET.SubElement(xml_root, "Domain")
        # mesh
        xml_toplogoy = ET.SubElement(xml_domain, "Topology",
                                     {'TopologyType': '3DRECTMesh',
                                      'Dimensions': "{0} {1} {2}".format(nz, ny, nx)})
        xml_geometry = ET.SubElement(xml_domain, 'Geometry',
                                     {'GeometryType': "VXVYVZ"})
        xml_geometry_x = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(nx),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_x.text = np.array_str(x).strip("[]").replace("\n", " ")
        xml_geometry_y = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(ny),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_y.text = np.array_str(y).strip("[]").replace("\n", " ")
        xml_geometry_z = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(nz),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_z.text = np.array_str(z).strip("[]").replace("\n", " ")

        # time card
        xml_time_grid = ET.SubElement(xml_domain, 'Grid',
                                      {'Name': 'TimeSeries',
                                       'GridType': 'Collection',
                                       'CollectionType': 'Temporal'})
        # loop over time
        for t_index in iseg:
            xml_itime_grid = ET.SubElement(xml_time_grid, "Grid",
                                           {'Name': str(round(da1["delta_day"][t_index], 3)),
                                            'GridType': 'Uniform'})
            xml_time_attr = ET.SubElement(
                xml_itime_grid, "Time",
                {'Value': str(round(da1["delta_day"][t_index], 3)),
                 "TimeType": "Single"})
            xml_topology_ref = ET.SubElement(
                xml_itime_grid, "Topology", {"Reference": "/Xdmf/Domain/Topology"})
            xml_geometry_ref = ET.SubElement(
                xml_itime_grid, "Geometry", {"Reference": "/Xdmf/Domain/Geometry"})
            xml_itime_temperature = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "Temperature",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_itime_temperature_dataitem = ET.SubElement(
                xml_itime_temperature, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_itime_temperature_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/"+str(round(da1["delta_day"][t_index], 3))+"/Temperature"

            xml_material = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "material",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_material_dataitem = ET.SubElement(
                xml_material, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_material_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Materials"

            xml_temperature_indicator = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "temperature_indicator",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_temperature_indicator_dataitem = ET.SubElement(
                xml_temperature_indicator, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_temperature_indicator_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Temperature_indicator"

            xml_thickness = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "thickness",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_thickness_dataitem = ET.SubElement(
                xml_thickness, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_thickness_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Thickness"

        # ouput xmdf
        with open(temperature_3d_xdmf+str(seg_index)+".xdmf", 'w') as f:
            f.write(prettify(xml_root))


def kriging_temperature():
    """
    krig temperature to the field.
    """

    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]
    ntherm = len(thermistors)
    therm_x = np.array([da1[x]["easting"] for x in thermistors])
    therm_y = np.array([da1[x]["northing"] for x in thermistors])
    therm_z = np.array([da1[x]["elevation"] for x in thermistors])
    therm_riverbed = np.array([da1[x]["riverbed"] for x in thermistors])
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_time = da1["time"]

    # truncate river data
    river_time = river["time"][(river["time"] >=
                                therm_time[0]) * (river["time"] <= therm_time[-1])]
    river_temp = river["temperature"][(river["time"] >=
                                       therm_time[0]) * (river["time"] <= therm_time[-1])]
    river_level = river["level"][(river["time"] >=
                                  therm_time[0]) * (river["time"] <= therm_time[-1])]

    # truncate thermistor data
    therm_data = therm_data[:, (therm_time >= river_time[0])
                            * (therm_time <= river_time[-1])]
    therm_time = therm_time[(therm_time >= river_time[0])
                            * (therm_time <= river_time[-1])]

    # down sample therm data
    therm_time = therm_time[np.arange(len(river_time))*3]
    therm_data = therm_data[:, np.arange(len(river_time))*3]

    # choose time segments (>30 days) with all thermistors functional
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm > 64)[0]
    valid_seg = np.where(np.diff(valid_index) > 1)[0]
    segment_start = valid_index[valid_seg+1]
    segment_end = valid_index[np.append(valid_seg[1:], -1)]
    segments = [np.arange(start, end)
                for start, end in zip(segment_start, segment_end)]
    segment_length = [len(x)*15/60/24 for x in segments]
    segments = [x for x, y in zip(segments, segment_length) if y > 30]
#    print([len(x)*15/60/24 for x in segments])

    segments = [segments[0][0:2], segments[1][0:2]]

    # read material file
    hdf5 = h5.File(material_h5, "r")
    material_ids = hdf5["Materials"][:].reshape(
        (nx, ny, material_nz), order="F")
    hdf5.close()
    # find riverbed
    river_bed = [[np.where(material_ids[ix, iy, :] == 1)[0][-1]
                  for iy in range(ny)] for ix in range(nx)]
    river_bed = [[x[ix], y[iy], material_z[river_bed[ix][iy]]+0.5*material_dz[river_bed[ix][iy]]]
                 for iy in range(ny) for ix in range(nx)]
    river_bed = np.array(river_bed)

    # use thermistor indicator to cut part of the thermistor rods
    # hardwired for visulization purpose
    material_ids = material_ids[:, :, -nz:]
    temperature_indicator = np.zeros(material_ids.shape)
    temperature_indicator[material_ids > 0] = 1
    thickest_z = np.max(np.sum(temperature_indicator, 2))
    x_index, y_index = np.where(np.sum(temperature_indicator, 2) == thickest_z)
    intersect_x = max(x_index)
    intersect_y = max(y_index)
    # x1, y1 = x[intersect_x], y[0]
    # x2, y2 = x[0], y[intersect_y]
    x1, y1 = 594474, y[0]
    x2, y2 = x[0], 116304
    west_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) < 0]
    y2 += 18*(y2-y1)/(x1-x2)
    x1 += 18
    east_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) > 0]
    blank_index = np.array(east_blank+west_blank)
    temperature_indicator[blank_index[:, 0], blank_index[:, 1], :] = 0

    # thickness of current location
    thickness = np.array([
        [(np.cumsum(dz*(material_ids[ix, iy, ::-1]))[::-1]-0.5*dz).tolist()
         for iy in range(ny)] for ix in range(nx)])
    # temperature_indicator = np.ones(nx*ny*nz)
    # # find locations near the thermistor (distance<4 m) in the model domain
    # xyz_flatten = np.array([[ix, iy, iz] for iz in z for iy in y for ix in x])
    # cells_near_therm = [np.where(((xyz_flatten[:, 0]-therm_x[i])**2 +
    #                               (xyz_flatten[:, 1]-therm_y[i])**2 +
    #                               (xyz_flatten[:, 2]-therm_z[i])**2*25) < 25)[0].tolist()
    #                     for i in range(ntherm)]
    # cells_near_therm = np.unique([x for y in cells_near_therm for x in y])
    # temperature_indicator[cells_near_therm] = 1
    # temperature_indicator = temperature_indicator.reshape(
    #     (nx, ny, nz), order="C")
    # hdf5.create_dataset("Temperature_indicator", data=temperature_indicator)
    hdf5 = h5.File(temperature_3d_h5, "w")
    hdf5.create_dataset("Materials", data=material_ids.flatten(order="F"))
    hdf5.create_dataset("Temperature_indicator",
                        data=temperature_indicator.flatten(order="F"))
    hdf5.create_dataset("Thickness",
                        data=thickness.flatten(order="F"))

    for iseg in segments:
        print(len(iseg))
        for t_index in iseg:
            print(t_index)
            # krig data for one segments
            riverbed_index = (therm_riverbed < river_level[t_index])
            input_x = np.append(therm_x, therm_x[riverbed_index])
            input_y = np.append(therm_y, therm_y[riverbed_index])
            input_z = np.append(therm_z, therm_riverbed[riverbed_index])
            input_data = np.append(therm_data[:, t_index],
                                   np.ones(np.sum(riverbed_index))*river_temp[t_index])
            uk3d = UniversalKriging3D(
                therm_x,
                therm_y,
                therm_z,
                therm_data[:, t_index],
                variogram_model='linear',
                drift_terms=['regional_linear'])
            k3d, ss3d = uk3d.execute('grid', x, y, z)
            # output data to hdf5
            temp_data = k3d.data.swapaxes(0, 2).flatten(order="F")
            group = hdf5.create_group(str(round(da1["delta_day"][t_index], 3)))
            group.create_dataset("Temperature", data=temp_data)
    hdf5.close()

    # write xdmf files
    for seg_index, iseg in enumerate(segments):
        print(len(iseg))

        xml_root = ET.Element("Xdmf", Version="3.0")
        xml_domain = ET.SubElement(xml_root, "Domain")
        # mesh
        xml_toplogoy = ET.SubElement(xml_domain, "Topology",
                                     {'TopologyType': '3DRECTMesh',
                                      'Dimensions': "{0} {1} {2}".format(nz, ny, nx)})
        xml_geometry = ET.SubElement(xml_domain, 'Geometry',
                                     {'GeometryType': "VXVYVZ"})
        xml_geometry_x = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(nx),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_x.text = np.array_str(x).strip("[]").replace("\n", " ")
        xml_geometry_y = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(ny),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_y.text = np.array_str(y).strip("[]").replace("\n", " ")
        xml_geometry_z = ET.SubElement(xml_geometry, 'DataItem',
                                       {'Dimensions': str(nz),
                                        "NumberType": "Float",
                                        "Precision": "8",
                                        "Format": "XML"})
        xml_geometry_z.text = np.array_str(z).strip("[]").replace("\n", " ")

        # time card
        xml_time_grid = ET.SubElement(xml_domain, 'Grid',
                                      {'Name': 'TimeSeries',
                                       'GridType': 'Collection',
                                       'CollectionType': 'Temporal'})
        # loop over time
        for t_index in iseg:
            xml_itime_grid = ET.SubElement(xml_time_grid, "Grid",
                                           {'Name': str(round(da1["delta_day"][t_index], 3)),
                                            'GridType': 'Uniform'})
            xml_time_attr = ET.SubElement(
                xml_itime_grid, "Time",
                {'Value': str(round(da1["delta_day"][t_index], 3)),
                 "TimeType": "Single"})
            xml_topology_ref = ET.SubElement(
                xml_itime_grid, "Topology", {"Reference": "/Xdmf/Domain/Topology"})
            xml_geometry_ref = ET.SubElement(
                xml_itime_grid, "Geometry", {"Reference": "/Xdmf/Domain/Geometry"})
            xml_itime_temperature = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "Temperature",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_itime_temperature_dataitem = ET.SubElement(
                xml_itime_temperature, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_itime_temperature_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/"+str(round(da1["delta_day"][t_index], 3))+"/Temperature"

            xml_material = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "material",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_material_dataitem = ET.SubElement(
                xml_material, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_material_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Materials"

            xml_temperature_indicator = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "temperature_indicator",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_temperature_indicator_dataitem = ET.SubElement(
                xml_temperature_indicator, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_temperature_indicator_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Temperature_indicator"

            xml_thickness = ET.SubElement(
                xml_itime_grid, "Attribute",
                {"Name": "thickness",
                 "AttributeType": "Scalar",
                 "Center": "Node"})
            xml_thickness_dataitem = ET.SubElement(
                xml_thickness, "DataItem",
                {"Format": "HDF",
                 "NumberType": "Float",
                 "Precision": "8",
                 "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
            xml_thickness_dataitem.text = temperature_3d_h5.split("/")[-1] + \
                ":/Thickness"

        # ouput xmdf
        with open(temperature_3d_xdmf+str(seg_index)+".xdmf", 'w') as f:
            f.write(prettify(xml_root))


def plot_bath_ringold_large():
    """
    read in data file and create material file
    """
    da1 = joblib.load(da1_joblib)

    # load unit data,create material
    unit_data = np.genfromtxt(geo_unit_file, skip_header=21)
    unit_x = np.sort(np.unique(unit_data[:, 0]))
    unit_y = np.sort(np.unique(unit_data[:, 1]))
    unit_ringold = unit_data[
        :, -1].reshape((len(unit_x), len(unit_y)), order="F")

    # load river bathmetry
    bath_data = np.transpose(np.genfromtxt(bathymetry_file, skip_header=6))
    bath_data = bath_data[:, ::-1]
    bath_data[bath_data < 0] = np.nan

    # plot bathmetry map
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]

    therm_x = np.array([da1[x]["easting"] for x in thermistors])
    therm_y = np.array([da1[x]["northing"] for x in thermistors])

    well_data = np.genfromtxt(well_file,
                              delimiter=",",
                              skip_header=1,
                              dtype="str")
    well_coord = dict()
    for iwell in well_data:
        if "399-" in iwell[0]:
            well_coord[iwell[0].split(
                "399-")[-1].lower()] = np.array(iwell[-2:][::-1], dtype="float")

    marked_well = ["2-1", "2-2", "2-3"]
    # check which bathmetry to update
    imgfile = img_dir+"large_ringold.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    cf = ax.contourf(unit_x,
                     unit_y,
                     np.transpose(unit_ringold),
                     levels=np.arange(88, 110.1, 0.1),
                     extend="both",
                     cmap=plt.cm.jet)

    iwell = list(well_coord.keys())[0]
    ax.scatter(well_coord[iwell][0],
               well_coord[iwell][1], marker="^",
               color="black", s=8, label="Wells")
    for iwell in list(well_coord.keys())[1:]:
        ax.scatter(well_coord[iwell][0],
                   well_coord[iwell][1],
                   marker="^",
                   color="black", s=8)
    for iwell in well_coord.keys():
        ax.text(well_coord[iwell][0]-30,
                well_coord[iwell][1]+20,
                iwell.upper(),
                fontsize=5.5)
    ax.scatter(therm_x,
               therm_y,
               c="black",
               label="Themistors",
               s=5)
    ax.add_patch(Rectangle(
        (ox, oy), ex-ox, ey-oy,
        fill=None,
        label="Dense array area",
        color="red", lw=2))
    ax4 = fig.add_axes([0.85, 0.06, 0.03, 0.9])
    cb = plt.colorbar(cf, cax=ax4, extend="both")
    cb.ax.set_ylabel("Bathymetry (m)",
                     rotation=270, labelpad=14)
    cb.set_ticks(np.arange(88, 110.2, 2))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_xlim([unit_x[0], unit_x[-1]])
    ax.set_ylim([unit_y[0], unit_y[-1]])
    ax.set_aspect(1)
    ax.legend(loc="upper right")
    fig.set_size_inches(7, 9)
    fig.subplots_adjust(left=0.1,
                        right=0.85,
                        bottom=0.06,
                        top=0.97,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"large_bath.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    cf = ax.contourf(bath_x,
                     bath_y,
                     np.transpose(bath_data),
                     levels=np.arange(103, 108.05, 0.05),
                     extend="both",
                     cmap=plt.cm.terrain)
    ax.scatter(therm_x,
               therm_y,
               c="black",
               label="Themistors",
               s=5)
    ax.add_patch(Rectangle(
        (ox, oy), ex-ox, ey-oy,
        fill=None,
        label="Dense array area",
        color="red", lw=2))
    ax4 = fig.add_axes([0.85, 0.06, 0.03, 0.9])
    cb = plt.colorbar(cf, cax=ax4, extend="both")
    cb.ax.set_ylabel("Bathymetry (m)",
                     rotation=270, labelpad=14)
    cb.set_ticks(np.arange(88, 110.2, 0.5))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect(1)
    ax.legend(loc="upper left")
    fig.set_size_inches(7, 9)
    fig.subplots_adjust(left=0.1,
                        right=0.85,
                        bottom=0.06,
                        top=0.97,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)


def truncated_data():

    da1 = joblib.load(da1_joblib)
    river = joblib.load(river_joblib)
    da1["delta_day"] = np.array(
        [(x-date_origin).total_seconds()/3600/24 for x in da1["time"]])

    # cut 2020 off
    #    date_end = datetime.strptime("2019-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    # read in data
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]
    ntherm = len(thermistors)

    ntherm = len(thermistors)
    therm_depth = [da1[x]["depth"] for x in thermistors]
    thermistors = [thermistors[x] for x in np.argsort(therm_depth)]
    therm_depth = [therm_depth[x] for x in np.argsort(therm_depth)]
    therm_elevation = [da1[x]["elevation"] for x in thermistors]
    therm_riverbed = [da1[x]["riverbed"] for x in thermistors]
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_delta_day = da1["delta_day"]
    therm_time = da1["time"]
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    gaps = np.where(valid_ntherm == 0)[0]
    truncate_start = therm_time[gaps][therm_time[gaps] < datetime.strptime(
        "2018-03-01 00:00:00", "%Y-%m-%d %H:%M:%S")][-1]
    truncate_end = therm_time[gaps][therm_time[gaps] > datetime.strptime(
        "2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")][0]

    # cut ending nan off
    therm_data = therm_data[:, (therm_time > truncate_start)
                            * (therm_time < truncate_end)]
    therm_delta_day = therm_delta_day[(therm_time > truncate_start)
                                      * (therm_time < truncate_end)]
    therm_time = therm_time[(therm_time > truncate_start)
                            * (therm_time < truncate_end)]

    # fill nan by simple linear interpolation
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm == ntherm)[0]
    therm_delta_day = therm_delta_day[:(valid_index[-1]+1)]
    therm_data = np.array([np.interp(therm_delta_day,
                                     therm_delta_day[valid_index],
                                     therm_data[i, valid_index])
                           for i in range(ntherm)])
    therm_data_nan = np.empty(therm_data.shape)
    therm_data_nan[:] = np.nan
    therm_data_nan[:, valid_index] = therm_data[:, valid_index]
    ntime = therm_data.shape[1]

    therm_low = np.nanmin(therm_data, 0)
    therm_high = np.nanmax(therm_data, 0)

    date_label = ["03/01",
                  "06/01",
                  "09/01",
                  "12/01"]
    date_label_loc = [datetime.strptime(
        x+"/2018", "%m/%y/%Y") for x in date_label]

    ncol = 10
    nrow = np.ceil(ntherm/ncol).astype(int)
    fig_name = img_dir + "da1_depth_2018.png"
    fig, axes = plt.subplots(nrow, ncol)
    for therm_index, ithermistor in enumerate(thermistors):
        ax = axes.flatten()[therm_index]
        ax.fill_between(therm_time, therm_low, therm_high,
                        color="lightgrey", label="Range")
        ax.plot(
            therm_time, therm_data[therm_index, :], color="red", label="Filled")
        ax.plot(
            therm_time,
            therm_data_nan[therm_index, :], color="black", label="Original")
        ax.set_xlabel('Date (2018)')
        ax.set_ylabel('Temperature $_oC$')
        ax.set_xticks(date_label_loc)
        ax.set_xticklabels(date_label)
        ax.set_ylim(0, 25)
        ax.set_title(
            "Depth ="+"{0:.3f}".format(da1[ithermistor]["depth"])+" m")
        ax.legend(loc="lower center", frameon=False)
    for ax in axes.flatten()[therm_index+1:ncol*nrow]:
        ax.set_axis_off()
    fig.set_size_inches(30, 20)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches=0)  # , transparent=True)
    plt.close(fig)
    print("Hello World!")

    # regualer dwt
    # wt_coeffs = [pywt.wavedec(therm_data[i, :], "db1", mode='sym',
    #                           level=None) for i in range(ntherm)]

    # wt_coeffs = [pywt.wavedec(therm_data[i, :], "db1",  # mode='sym',
    #                           level=15) for i in range(ntherm)]
    # wt_rec = [[pywt.upcoef("a", y, "db1", level=len(x))[:ntime] if i == 0 else
    #            pywt.upcoef("d", y, "db1", level=len(x)+1-i)[:ntime]
    #            for i, y in enumerate(x)] for x in wt_coeffs]
    # for itherm in range(ntherm):
    #     imgfile = img_dir+"temp/"+str(itherm)+".png"
    #     fig = plt.figure()
    #     ax = plt.subplot(111)
    #     xxx = np.zeros(ntime)
    #     for x in wt_rec[itherm]:
    #         ax.plot(therm_time, x)
    #         xxx += x
    #     ax.plot(therm_time, xxx, color="black")
    #     fig.set_size_inches(16, 4)
    #     fig.savefig(imgfile, bbox_inches=0, dpi=300)
    #     plt.close(fig)
    truncated_da1 = dict()
    truncated_da1["time"] = therm_time
    truncated_da1["data"] = therm_data
    truncated_da1["thermistors"] = thermistors
    truncated_da1["northing"] = [da1[x]["northing"] for x in thermistors]
    truncated_da1["elevation"] = [da1[x]["elevation"] for x in thermistors]
    truncated_da1["easting"] = [da1[x]["easting"] for x in thermistors]
    truncated_da1["riverbed"] = [da1[x]["riverbed"] for x in thermistors]
    truncated_da1["depth"] = [da1[x]["depth"] for x in thermistors]
    joblib.dump(truncated_da1, truncated_da1_joblib)


def paraview_mean_temp():
        # read material file
    hdf5 = h5.File(material_h5, "r")
    material_ids = hdf5["Materials"][:].reshape(
        (nx, ny, material_nz), order="F")
    hdf5.close()

    # use thermistor indicator to cut part of the thermistor rods
    # hardwired for visulization purpose
    material_ids = material_ids[:, :, -nz:]
    temperature_indicator = np.zeros(material_ids.shape)
    temperature_indicator[material_ids > 0] = 1
    thickest_z = np.max(np.sum(temperature_indicator, 2))
    x_index, y_index = np.where(
        np.sum(temperature_indicator, 2) == thickest_z)
    intersect_x = max(x_index)
    intersect_y = max(y_index)
    # x1, y1 = x[intersect_x], y[0]
    # x2, y2 = x[0], y[intersect_y]
    x1, y1 = 594474, y[0]
    x2, y2 = x[0], 116304
    west_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) < 0]
    y2 += 18*(y2-y1)/(x1-x2)
    x1 += 18
    east_blank = [[ix, iy]
                  for ix in range(nx) for iy in range(ny) if
                  ((x[ix]-x1)*(y2-y1)-(y[iy]-y1)*(x2-x1)) > 0]
    blank_index = np.array(east_blank+west_blank)
    temperature_indicator[blank_index[:, 0], blank_index[:, 1], :] = 0

    # thickness of current location
    thickness = np.array([
        [(np.cumsum(dz*(material_ids[ix, iy, ::-1]))[::-1]-0.5*dz).tolist()
         for iy in range(ny)] for ix in range(nx)])

    hdf5 = h5.File(mean_temperature_h5, "w")
    hdf5.create_dataset("Materials", data=material_ids.flatten(order="F"))
    hdf5.create_dataset("Temperature_indicator",
                        data=temperature_indicator.flatten(order="F"))
    hdf5.create_dataset("Thickness",
                        data=thickness.flatten(order="F"))

    # mean temperature
    uk3d = UniversalKriging3D(
        easting,
        northing,
        elevation,
        mean_temp,
        variogram_model='linear',
        drift_terms=['regional_linear'])
    k3d, ss3d = uk3d.execute('grid', x, y, z)
    temp_data = k3d.data.swapaxes(0, 2).flatten(order="F")
    hdf5.create_dataset("Mean_temperature",
                        data=temp_data.flatten(order="F"))

    # plot std temperature
    uk3d = UniversalKriging3D(
        easting,
        northing,
        elevation,
        std_temp,
        variogram_model='linear',
        drift_terms=['regional_linear'])
    k3d, ss3d = uk3d.execute('grid', x, y, z)
    temp_data = k3d.data.swapaxes(0, 2).flatten(order="F")
    hdf5.create_dataset("Std_temperature",
                        data=temp_data.flatten(order="F"))
    hdf5.close()

    xml_root = ET.Element("Xdmf", Version="3.0")
    xml_domain = ET.SubElement(xml_root, "Domain")
    # mesh
    xml_toplogoy = ET.SubElement(xml_domain, "Topology",
                                 {'TopologyType': '3DRECTMesh',
                                  'Dimensions': "{0} {1} {2}".format(nz, ny, nx)})
    xml_geometry = ET.SubElement(xml_domain, 'Geometry',
                                 {'GeometryType': "VXVYVZ"})
    xml_geometry_x = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(nx),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_x.text = np.array_str(x).strip("[]").replace("\n", " ")
    xml_geometry_y = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(ny),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_y.text = np.array_str(y).strip("[]").replace("\n", " ")
    xml_geometry_z = ET.SubElement(xml_geometry, 'DataItem',
                                   {'Dimensions': str(nz),
                                    "NumberType": "Float",
                                    "Precision": "8",
                                    "Format": "XML"})
    xml_geometry_z.text = np.array_str(z).strip("[]").replace("\n", " ")

    xml_grid = ET.SubElement(xml_domain, "Grid",
                             {'Name': "Statistics",
                              'GridType': 'Uniform'})
    xml_topology_ref = ET.SubElement(
        xml_grid, "Topology", {"Reference": "/Xdmf/Domain/Topology"})
    xml_geometry_ref = ET.SubElement(
        xml_grid, "Geometry", {"Reference": "/Xdmf/Domain/Geometry"})

    xml_mean_temperature = ET.SubElement(
        xml_grid, "Attribute",
        {"Name": "Mean temperature",
         "AttributeType": "Scalar",
         "Center": "Node"})
    xml_mean_temperature_dataitem = ET.SubElement(
        xml_mean_temperature, "DataItem",
        {"Format": "HDF",
         "NumberType": "Float",
         "Precision": "8",
         "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
    xml_mean_temperature_dataitem.text = mean_temperature_h5.split("/")[-1] + \
        ":/Mean_temperature"

    xml_std_temperature = ET.SubElement(
        xml_grid, "Attribute",
        {"Name": "Std temperature",
         "AttributeType": "Scalar",
         "Center": "Node"})
    xml_std_temperature_dataitem = ET.SubElement(
        xml_std_temperature, "DataItem",
        {"Format": "HDF",
         "NumberType": "Float",
         "Precision": "8",
         "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
    xml_std_temperature_dataitem.text = mean_temperature_h5.split("/")[-1] + \
        ":/Std_temperature"

    xml_material = ET.SubElement(
        xml_grid, "Attribute",
        {"Name": "material",
         "AttributeType": "Scalar",
         "Center": "Node"})
    xml_material_dataitem = ET.SubElement(
        xml_material, "DataItem",
        {"Format": "HDF",
         "NumberType": "Float",
         "Precision": "8",
         "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
    xml_material_dataitem.text = mean_temperature_h5.split("/")[-1] + \
        ":/Materials"

    xml_temperature_indicator = ET.SubElement(
        xml_grid, "Attribute",
        {"Name": "temperature_indicator",
         "AttributeType": "Scalar",
         "Center": "Node"})
    xml_temperature_indicator_dataitem = ET.SubElement(
        xml_temperature_indicator, "DataItem",
        {"Format": "HDF",
         "NumberType": "Float",
         "Precision": "8",
         "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
    xml_temperature_indicator_dataitem.text = mean_temperature_h5.split("/")[-1] + \
        ":/Temperature_indicator"

    xml_thickness = ET.SubElement(
        xml_grid, "Attribute",
        {"Name": "thickness",
         "AttributeType": "Scalar",
         "Center": "Node"})
    xml_thickness_dataitem = ET.SubElement(
        xml_thickness, "DataItem",
        {"Format": "HDF",
         "NumberType": "Float",
         "Precision": "8",
         "Dimensions": "{0} {1} {2}".format(nz, ny, nx)})
    xml_thickness_dataitem.text = mean_temperature_h5.split("/")[-1] + \
        ":/Thickness"

    # ouput xmdf
    with open(mean_temperature_xdmf, 'w') as f:
        f.write(prettify(xml_root))


def plot_mean_variance():

    # load data
    da1 = joblib.load(truncated_da1_joblib)
    river = joblib.load(river_joblib)
    air = joblib.load(air_joblib)

    # read value from dict
    northing = np.array(da1["northing"])
    easting = np.array(da1["easting"])
    riverbed = np.array(da1["riverbed"])
    elevation = np.array(da1["elevation"])
    depth = np.array(da1["depth"])
    thermistors = np.array(da1["thermistors"])
    time = np.array(da1["time"])
    data = np.array(da1["data"])

    # make sure river and themistor uses the same periods
    time = river["time"][(river["time"] >= da1["time"][0]) *
                         (river["time"] <= da1["time"][-1])]
    river_temperature = river["temperature"][(river["time"] >= time[0]) *
                                             (river["time"] <= time[-1])]
    air_temperature = air["temperature"][(air["time"] >= time[0]) *
                                         (air["time"] <= time[-1])]
    air_temperature = np.interp(
        np.arange(len(time)),
        np.arange(len(time))[~np.isnan(air_temperature)],
        air_temperature[~np.isnan(air_temperature)])
    data = np.array(da1["data"])[:, find_match(time, da1["time"])[-1]]

    # mean thermistor
    mean_temp = np.mean(data, 1)
    std_temp = np.std(data, 1)
    cv_temp = std_temp/mean_temp

    # mean river
    mean_river = np.mean(river_temperature)
    std_river = np.std(river_temperature)
    cv_river = std_river/mean_river

    # mean air
    mean_air = np.mean(air_temperature)
    std_air = np.std(air_temperature)
    cv_air = std_air/mean_air

    # locate groups
    riverbed_group = [np.where(riverbed < 104.2)[0],
                      np.where((riverbed > 104.2)*(riverbed < 105.2))[0],
                      np.where(riverbed > 105.2)[0]]
    riverbed_group_names = ["104 m", "104.5 m", "105.5 m"]
    riverbed_group_color = ["green", "black", "red"]

    imgfile = img_dir+"scatter/VS_mean_temp_std_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(mean_river, std_river,
               edgecolor="blue",
               facecolor="blue",
               s=30,
               marker="^",
               label="River")
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(mean_temp[igroup], std_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Mean temperature ($^o$C)")
    ax.set_ylabel("SD of temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_mean_temp_std_temp_with_air.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(mean_air, std_air,
               edgecolor="Orange",
               facecolor="Orange",
               s=30,
               marker="*",
               label="Air")
    ax.scatter(mean_river, std_river,
               edgecolor="blue",
               facecolor="blue",
               s=30,
               marker="^",
               label="River")
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(mean_temp[igroup], std_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Mean temperature ($^o$C)")
    ax.set_ylabel("SD of temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_elevation_mean_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(elevation[igroup], mean_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor elevation (m)")
    ax.set_ylabel("Mean temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_elevation_std_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(elevation[igroup], std_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor elevation (m)")
    ax.set_ylabel("SD of temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_depth_mean_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(depth[igroup], mean_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor depth (m)")
    ax.set_ylabel("Mean temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_depth_std_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(depth[igroup], std_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor depth (m)")
    ax.set_ylabel("SD of temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_riverbed_mean_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(riverbed[igroup], mean_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor riverbed (m)")
    ax.set_ylabel("Mean temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    imgfile = img_dir+"scatter/VS_riverbed_std_temp.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    for group_index, igroup in enumerate(riverbed_group):
        ax.scatter(riverbed[igroup], std_temp[igroup],
                   edgecolor=riverbed_group_color[group_index],
                   facecolor="none",
                   s=30,
                   label=riverbed_group_names[group_index])
    ax.set_xlabel("Thermistor riverbed (m)")
    ax.set_ylabel("SD of temperature ($^o$C)")
    ax.legend()
    fig.set_size_inches(4, 3.5)
    fig.subplots_adjust(left=0.2,
                        right=0.9,
                        bottom=0.15,
                        top=0.95,
                        wspace=0.25,
                        hspace=0.3)
    fig.savefig(imgfile, bbox_inches=0, dpi=300)
    plt.close(fig)

    # paraview_mean_temp()


def plot_air():
    # plot air temperture time series
    air = joblib.load(air_joblib)
    imgfile = img_dir+"air_temperature.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(air["time"],
            air["temperature"],
            lw=0.5,
            color="orange", label="Air")
    ax.set_ylabel('Temperature ($^oC$)')
    ax.set_xlim(air["time"][0], air["time"][-1])
#    ax.set_ylim(104, 109.5)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    fig.set_size_inches(10, 3.5)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)


def plot_river_well():
    # plot rive well time series
    da1 = joblib.load(da1_joblib)
    da1["delta_day"] = np.array(
        [(x-date_origin).total_seconds()/3600/24 for x in da1["time"]])
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])

    therm_low = np.nanmin(therm_data, 0)
    therm_high = np.nanmax(therm_data, 0)
    # modify to remove abnormal peak! hard wired
    start = np.where((da1["time"] < datetime.strptime(
        "2017-08-01", "%Y-%m-%d"))*(therm_high > 18))[0][0]
    end = start+10000
    therm_high[start:end] = np.interp(da1["delta_day"][start:end],
                                      [da1["delta_day"][start],
                                       da1["delta_day"][end]],
                                      [therm_high[start], 22])
#                                       therm_high[end]

    truncated_da1 = joblib.load(truncated_da1_joblib)
    river = joblib.load(river_joblib)
    well2_1 = joblib.load(well2_1_joblib)
    well2_2 = joblib.load(well2_2_joblib)
    well2_3 = joblib.load(well2_3_joblib)

    imgfile = img_dir+"river_groundwater_level_long.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(river["time"][np.arange(int(len(river["time"])/3))*3],
            river["level"][np.arange(int(len(river["time"])/3))*3],
            lw=0.5,
            color="blue", label="River")
    ax.plot(well2_1["time"][np.arange(int(len(well2_1["time"])/3))*3],
            well2_1["level"][np.arange(int(len(well2_1["time"])/3))*3],
            lw=0.5,
            color="green", label="Well 2-1")
    ax.plot(well2_2["time"][np.arange(int(len(well2_2["time"])/3))*3],
            well2_2["level"][np.arange(int(len(well2_2["time"])/3))*3],
            lw=0.5,
            color="orange", label="Well 2-2")
    ax.plot(well2_3["time"][np.arange(int(len(well2_3["time"])/3))*3],
            well2_3["level"][np.arange(int(len(well2_3["time"])/3))*3],
            lw=0.5,
            color="red", label="Well 2-3")
    ax.set_xlabel('Date')
    ax.set_ylabel('Water level (m)')
    ax.set_xlim(river["time"][0], river["time"][-1])
    ax.set_ylim(104, 109.5)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    fig.set_size_inches(10, 3.5)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)

    imgfile = img_dir+"river_groundwater_temperature_long.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.fill_between(da1["time"], therm_low, therm_high,
                    color="lightgrey", label="Range of thermistors")
    ax.plot(river["time"][np.arange(int(len(river["time"])/3))*3],
            river["temperature"][np.arange(int(len(river["time"])/3))*3],
            lw=0.5,
            color="blue", label="River")
    ax.plot(well2_1["time"][np.arange(int(len(well2_1["time"])/3))*3],
            well2_1["temperature"][np.arange(int(len(well2_1["time"])/3))*3],
            lw=0.5,
            color="green", label="Well 2-1")
    ax.plot(well2_2["time"][np.arange(int(len(well2_2["time"])/3))*3],
            well2_2["temperature"][np.arange(int(len(well2_2["time"])/3))*3],
            lw=0.5,
            color="orange", label="Well 2-2")
    ax.plot(well2_3["time"][np.arange(int(len(well2_3["time"])/3))*3],
            well2_3["temperature"][np.arange(int(len(well2_3["time"])/3))*3],
            lw=0.5,
            color="red", label="Well 2-3")
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature ($^oC$)')
    ax.set_xlim(river["time"][0], river["time"][-1])
    ax.set_ylim(0, 25)
    ax.legend(loc="upper left", frameon=False, ncol=5)
    fig.set_size_inches(10, 3.5)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)

    date_label = ["03/01",
                  "06/01",
                  "09/01",
                  "12/01"]
    date_label_loc = [datetime.strptime(
        x+"/2018", "%m/%y/%Y") for x in date_label]

    imgfile = img_dir+"river_groundwater_level_short.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(river["time"][np.arange(int(len(river["time"])/3))*3],
            river["level"][np.arange(int(len(river["time"])/3))*3],
            lw=0.5,
            color="blue", label="River")
    ax.plot(well2_1["time"][np.arange(int(len(well2_1["time"])/3))*3],
            well2_1["level"][np.arange(int(len(well2_1["time"])/3))*3],
            lw=0.5,
            color="green", label="Well 2-1")
    ax.plot(well2_2["time"][np.arange(int(len(well2_2["time"])/3))*3],
            well2_2["level"][np.arange(int(len(well2_2["time"])/3))*3],
            lw=0.5,
            color="orange", label="Well 2-2")
    ax.plot(well2_3["time"][np.arange(int(len(well2_3["time"])/3))*3],
            well2_3["level"][np.arange(int(len(well2_3["time"])/3))*3],
            lw=0.5,
            color="red", label="Well 2-3")
    ax.set_xlabel('Date')
    ax.set_ylabel('Water level (m)')
    ax.set_xticks(date_label_loc)
    ax.set_xticklabels(date_label)
    ax.set_xlim(truncated_da1["time"][0], truncated_da1["time"][-1])
    ax.set_ylim(104, 109.5)
    ax.legend(loc="upper right", frameon=False)
    fig.set_size_inches(5, 3.5)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)

    therm_low = np.nanmin(truncated_da1["data"], 0)
    therm_high = np.nanmax(truncated_da1["data"], 0)

    imgfile = img_dir+"river_groundwater_temperature_short.png"
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.fill_between(truncated_da1["time"], therm_low, therm_high,
                    color="lightgrey", label="Range of thermistors")
    ax.plot(river["time"][np.arange(int(len(river["time"])/3))*3],
            river["temperature"][np.arange(int(len(river["time"])/3))*3],
            lw=0.5,
            color="blue", label="River")
    ax.plot(well2_1["time"][np.arange(int(len(well2_1["time"])/3))*3],
            well2_1["temperature"][np.arange(int(len(well2_1["time"])/3))*3],
            lw=0.5,
            color="green", label="Well 2-1")
    ax.plot(well2_2["time"][np.arange(int(len(well2_2["time"])/3))*3],
            well2_2["temperature"][np.arange(int(len(well2_2["time"])/3))*3],
            lw=0.5,
            color="orange", label="Well 2-2")
    ax.plot(well2_3["time"][np.arange(int(len(well2_3["time"])/3))*3],
            well2_3["temperature"][np.arange(int(len(well2_3["time"])/3))*3],
            lw=0.5,
            color="red", label="Well 2-3")
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature ($^oC$)')
    ax.set_xticks(date_label_loc)
    ax.set_xticklabels(date_label)
    ax.set_xlim(truncated_da1["time"][0], truncated_da1["time"][-1])
    ax.set_ylim(0, 25)
    ax.legend(loc="lower center", frameon=False)
    fig.set_size_inches(5, 3.5)
    fig.tight_layout()
    fig.savefig(imgfile, bbox_inches=0, dpi=300)


def modwt():

    da1 = joblib.load(truncated_da1_joblib)
    time = np.array(da1["time"])
    data = np.array(da1["data"])
    northing = np.array(da1["northing"])
    easting = np.array(da1["easting"])
    riverbed = np.array(da1["riverbed"])
    elevation = np.array(da1["elevation"])
    depth = np.array(da1["depth"])
    thermistors = np.array(da1["thermistors"])
    ntherm = len(thermistors)

    # data need to be pad
    pad_head = int((2**17-len(data[0, :]))/2)
    pad_tail = 2**17-len(data[0, :])-int((2**17-len(data[0, :]))/2)

    # caculate coef
    modwt_coef = dict()
    for therm_index, itherm in enumerate(thermistors):
        print(itherm)
        pad_data = np.pad(data[therm_index, :],
                          (pad_head, pad_tail),
                          mode="symmetric")
        modwt_coef[itherm] = pywt.swt(pad_data, "db1", level=None, start_level=0,
                                      trim_approx=True, norm=True)
    joblib.dump(modwt_coef, results_dir+"modwt_coef.joblib")

    # caculate variance
    modwt_var = dict()
    modwt_var_mean = dict()
    for therm_index, itherm in enumerate(thermistors):
        print(itherm)
        modwt_var[itherm] = []
        modwt_var_mean[itherm] = []
        for ilevel in range(len(modwt_coef[itherm]))[1:]:
            modwt_var_level = modwt_coef[itherm][ilevel]**2
            modwt_var_level = modwt_var_level[pad_head:-pad_tail]
            modwt_var[itherm].append(modwt_var_level)
            modwt_var_mean[itherm].append(np.nanmean(modwt_var_level))
    joblib.dump(modwt_var, results_dir+"modwt_var.joblib")
    joblib.dump(modwt_var_mean, results_dir+"modwt_var_mean.joblib")

    fig_name = img_dir + "modwt.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ithermistor in thermistors:
        ax.plot(modwt_var_mean[ithermistor][::-1])
    fig.set_size_inches(5, 4)
    fig.savefig(fig_name, dpi=300, transparent=False)


def plot_river_well_cwt():
    """
    plot dev
    """
    river = joblib.load(river_joblib)
    well2_3 = joblib.load(well2_3_joblib)

    # define arange of data
    date_start = datetime.strptime("2017-02-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    date_end = datetime.strptime("2019-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    # caculate wavelet coef
    river_level = river["level"][
        (river["time"] >= date_start) * (river["time"] <= date_end)]
    river_temperature = river["temperature"][
        (river["time"] >= date_start) * (river["time"] <= date_end)]
    river_time = river["time"][(
        river["time"] >= date_start) * (river["time"] <= date_end)]
    well2_3_level = well2_3["level"][
        (well2_3["time"] >= date_start) * (well2_3["time"] <= date_end)]
    well2_3_temperature = well2_3["temperature"][
        (well2_3["time"] >= date_start) * (well2_3["time"] <= date_end)]
    well2_3_time = well2_3["time"][(
        well2_3["time"] >= date_start) * (well2_3["time"] <= date_end)]

    # number of scale
    nscale = 200

    # river level
    river_level_coef = dense_array_cwt(river_time, river_level, nscale)
    fig_name = img_dir + "cwt/river_level_long.png"
    plot_cwt_long(fig_name, river_level_coef)

    # river temperature
    river_temperature_coef = dense_array_cwt(
        river_time, river_temperature, nscale)
    fig_name = img_dir + "cwt/river_temperature_long.png"
    plot_cwt_long(fig_name, river_temperature_coef)

    # well2_3 level
    well2_3_level_coef = dense_array_cwt(
        well2_3_time, well2_3_level, nscale)
    fig_name = img_dir + "cwt/well2_3_level_long.png"
    plot_cwt_long(fig_name, well2_3_level_coef)

    # well2_3 temperature
    well2_3_temperature_coef = dense_array_cwt(
        well2_3_time, well2_3_temperature, nscale)
    fig_name = img_dir + "cwt/well2_3_temperature_long.png"
    plot_cwt_long(fig_name, well2_3_temperature_coef)


def plot_thermistor_cwt():
    """
    plot cwt of thermistors
    """

    # load data
    thermistors = joblib.load(truncated_da1_joblib)
    river = joblib.load(river_joblib)
    well2_3 = joblib.load(well2_3_joblib)

    # clean data
    time = river["time"][(river["time"] >= thermistors["time"][0]) *
                         (river["time"] <= thermistors["time"][-1])]
    river_level = river["level"][(river["time"] >= time[0]) *
                                 (river["time"] <= time[-1])]
    well_level = well2_3["level"][(well2_3["time"] >= time[0]) *
                                  (well2_3["time"] <= time[-1])]
    river_temperature = river["temperature"][(river["time"] >= time[0]) *
                                             (river["time"] <= time[-1])]
    well_temperature = well2_3["temperature"][(well2_3["time"] >= time[0]) *
                                              (well2_3["time"] <= time[-1])]
    thermistor_data = thermistors["data"][:,
                                          find_match(time, thermistors["time"])[-1]]

    nscale = 200
    # plot well
    well_cwt = dense_array_cwt(
        time, well_temperature, nscale)
    fig_name = img_dir + "cwt/short/"+"well.png"
    plot_cwt_short(fig_name, well_cwt, "Well")

    # plot river
    river_cwt = dense_array_cwt(
        time, river_temperature, nscale)
    fig_name = img_dir + "cwt/short/"+"river.png"
    plot_cwt_short(fig_name, river_cwt, "River")

    # plot well
    well_cwt = dense_array_cwt(
        time, well_level, nscale)
    fig_name = img_dir + "cwt/short/"+"well_level.png"
    plot_cwt_short(fig_name, well_cwt, "Well")

    # plot river
    river_cwt = dense_array_cwt(
        time, river_level, nscale)
    fig_name = img_dir + "cwt/short/"+"river_level.png"
    plot_cwt_short(fig_name, river_cwt, "River")

    for thermistor_index, ithermistor in enumerate(thermistors["thermistors"]):
        print(ithermistor)
        therm_cwt = dense_array_cwt(
            time, thermistor_data[thermistor_index, :], nscale)
        fig_name = img_dir + "cwt/short/"+ithermistor+".png"
        title = ("Elevation = " +
                 str(np.round(thermistors["elevation"][thermistor_index], 3)) +
                 " m; Depth = " +
                 str(np.round(thermistors["depth"][thermistor_index], 3))+" m")
        plot_cwt_short(fig_name, therm_cwt, title)


def tobe():
    cwt1 = dense_array_cwt(time, river_level, nscale)
    cwt2 = dense_array_cwt(time, well_level, nscale)
    coef1 = cwt1["coef"]
    coef2 = cwt2["coef"]
    coef12 = coef1*np.conj(coef2)
    wcoh = coef12**2/(coef1*coef2)

    # # save pt status
    # pt_fname = pt_dir+"cwt_stage.pk"
    # file = open(pt_fname, "wb")
    # pickle.dump(cwt_stage, file)
    # file.close()
    # with open(pt_dir+"cwt_stage.pk", "rb") as fname:
    #     cwt_stage = pickle.load(fname)
    # ax.set_xticks(year_label_loc/24)
    # ax.set_xticklabels(year_label)
