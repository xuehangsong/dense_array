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
import copy
import csv
from sklearn.externals import joblib
import mpl_scatter_density
import matplotlib.pyplot as plt
# from fastdtw import fastdtw
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
import h5py as h5
from scipy.interpolate import RectBivariateSpline
from xml.etree import ElementTree as ET
from xml.dom import minidom
from scipy.interpolate import interp2d
from pykrige.uk3d import UniversalKriging3D
from pykrige.uk import UniversalKriging
from matplotlib.patches import Rectangle
from pykrige.uk3d import UniversalKriging3D
import multiprocessing as mp
import pywt


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

# load preprocessed data
da1 = joblib.load(da1_joblib)

date_origin = datetime.strptime("2017-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
da1["delta_day"] = np.array(
    [(x-date_origin).total_seconds()/3600/24 for x in da1["time"]])
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
bath_x = np.arange(593000.5598+0.5, 594900.5598, 1)
bath_y = np.arange(114500.6771+0.5, 117800.6771, 1)


def plot_depth_sorted():
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
    fig_name = img_dir + "da1_depth_sorted.png"
    fig, axes = plt.subplots(nrow, ncol)
    for therm_index, ithermistor in enumerate(thermistors):
        ax = axes.flatten()[therm_index]
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


def plot_mean_variance():
    """
    plot elevation sorted by thermistor depth
    """

    date_end = datetime.strptime("2019-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")

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
    therm_data = therm_data[:, da1["time"] < date_end]
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
            da1["time"], da1[ithermistor]["temperature"], color="blue")

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


def plot_bath_ringold_large():
    """
    read in data file and create material file
    """
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


def wavelet():

    # cut 2020 off
    date_end = datetime.strptime("2019-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")

    # read in data
    thermistors = [x for x in list(
        da1.keys()) if x != "time" and x != "delta_day"]
    ntherm = len(thermistors)
    therm_elevation = [da1[x]["elevation"] for x in thermistors]
    therm_depth = [da1[x]["depth"] for x in thermistors]
    therm_riverbed = [da1[x]["riverbed"] for x in thermistors]
    therm_data = np.array([da1[x]["temperature"] for x in thermistors])
    therm_delta_day = da1["delta_day"]
    therm_time = da1["time"]

    # cut ending nan off
    therm_data = therm_data[:, therm_time < date_end]
    therm_delta_day = therm_delta_day[therm_time < date_end]
    therm_time = therm_time[therm_time < date_end]

    # fill nan by simple linear interpolation
    valid_ntherm = np.count_nonzero(~np.isnan(therm_data), axis=0)
    valid_index = np.where(valid_ntherm == ntherm)[0]
    therm_time = therm_time[:(valid_index[-1]+1)]
    therm_delta_day = therm_delta_day[:(valid_index[-1]+1)]
    therm_data = therm_data[:, :(valid_index[-1]+1)]
    therm_data = np.array([np.interp(therm_delta_day, therm_delta_day[valid_index], therm_data[i, valid_index])
                           for i in range(ntherm)])
    ntime = therm_data.shape[1]

    wt_coeffs = [pywt.wavedec(therm_data[i, :], "db1", mode='sym',
                              level=None) for i in range(ntherm)]

    # wt_coeffs_single = [[[None]*i + [x]+[None]*(len(itherm)-i-1)
    #                      for i, x in enumerate(itherm)] for itherm in wt_coeffs]

    wt_rec = [[[None]*i + [x]+[None]*(len(itherm)-i-1)
               for i, x in enumerate(itherm)] for itherm in wt_coeffs]

   # xxx = [[pywt.waverec(x, "db1") for x in itherm]
   #         for itherm in wt_coeffs_single]

    [wt_coeffs[0], "db1"]
    # wavelet_data = signal.detrend(river_level[:, 3])
    # wavelet_time = river_level[:, 0]

    # # 1/pywt.scale2frequency("morl",[0.40625,1.95e4])
    # scale_lower = 0.40625
    # scale_upper = 1.95e4
    # scale_n_log = 201
    # scale_log_base = (scale_upper/scale_lower)**(1/scale_n_log)
    # scale = scale_lower*scale_log_base**(np.arange(scale_n_log+1))
    # coef, freq = pywt.cwt(
    #     wavelet_data,  scale, "morl", 3600)
    # power = (np.abs(coef))**2/(scale[:, None])
    # period = 1/freq/3600/24

    # # scale = 2**np.arange(17)
    # # coef, freq = pywt.cwt(
    # #     wavelet_data,  scale, "morl", 3600)

    # # power = (abs(coef))**2/(scale[:, None])
    # # period = 1/freq/3600/24

    # # put data in pickle file
    # cwt_stage = dict()
    # cwt_stage["time"] = wavelet_time
    # cwt_stage["freq"] = freq
    # cwt_stage["scale"] = scale
    # cwt_stage["coef"] = coef
    # cwt_stage["power"] = power
    # cwt_stage["period"] = period

    # # save pt status
    # pt_fname = case_dir+"cwt_stage.pk"
    # file = open(pt_fname, "wb")
    # pickle.dump(cwt_stage, file)
    # file.close()
