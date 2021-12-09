from astropy.coordinates.sky_coordinate import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import astropy
from astropy import coordinates as coords
from astropy import units 
from astropy.units import cds
# import cartopy.crs as ccrs
import HI_statistics as HI
import multicurvefit as mcfit

from colour import Color

from scipy import optimize
from scipy import stats

import typing as typ
import os
from os.path import isfile, join
import json
from scipy import signal


V0 = 220 # km/s
R0 = 8.5 # kpc


def rightmost_peak(data, key):
    v_rel = data[key]['v_rel']
    amp = data[key]['amp']
    (x, y) = signal.find_peaks(amp, height=30)
    print(x)
    print(y)
    y = y['peak_heights']
    return x[-1]

def lon(data):
    keys = list(data)
    lons = []
    for key in keys:
        lons.append(data[key]['l'])
    return lons

def radius(data):
    keys = list(data)
    radiuss = []
    for key in keys:
        l = data[key]['l']
        r = R0 * np.sin(l)
        radiuss.append(np.abs(r))
    return radiuss

def velocity(data):
    keys = list(data)
    vels = []
    for key in keys:
        v = rightmost_peak(data, key)
        vels.append(v)
    return vels

def rot_array(data):
    rot_rs = []
    rot_vs = []
    for key in list(data):
        l = data[key]['l']
        b = data[key]['b']
        if l < 90 or l > 180 or b > 1 or b < -1:
            continue
        print(key)
        r = R0 * np.sin(l)
        v = rightmost_peak(data,key)
        rot_rs.append(np.abs(r))
        rot_vs.append(v)
    return rot_rs, rot_vs

def maximum(data):
    l = data['l']
    b = data['b']
    v_rel = data['v_rel']
    amp = data['amp']
    ix_max = np.argmax(amp)
    v_max = v_rel[ix_max]
    return l, b, v_max 

def plot_v_map(data):

    cfunc = lambda x: cm.get_cmap('inferno')(x/5)
    colors = []
    for key in list(data):
        max_v = data[key]['v_rel'][np.argmax(data[key]['amp'])]
        colors.append(cfunc(max_v))
    x = []
    y = []
    for key in list(data):
        x.append(data[key]['l'])
        y.append(data[key]['b'])
    eq = coords.SkyCoord(x, y, frame='galactic', unit=units.deg)
    gal = eq.galactic
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="mollweide")
    plt.grid(True)
    ax.scatter(gal.l.wrap_at('180d').radian, gal.b.radian,c=colors)

def find_max_v(data):
    keys = list(data)
    max_ixs = []
    for key in keys:
        max_ixs.append(np.argmax(data[key]['amp']))
    return np.max(max_ixs)

def plot_v_lon(data):
    keys = list(data)
    lons = []
    lats = []
    max_vs = []
    for key in keys:
        lons.append(data[key]['l'])
        lats.append(data[key]['b'])
        max_vs.append(data[key]['v_rel'][np.argmax(data[key]['amp'])])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(max_vs,lons,lats,alpha=0.5)
    plt.xlim(0,360)
    ax.set_ylabel("Longitude (deg)")
    ax.set_zlabel("Latitude (deg)")
    ax.set_zlabel(r"$v_{\mathrm{rel}}$")
    
    plt.show()
    

d_struct = []
with open("hi_data.json") as fp:
    d_struct = json.load(fp)
keys = list(d_struct)
test_data = d_struct[keys[0]]


# plot_v_map(d_struct)
# plt.show()

plot_v_lon(d_struct)

# v_rel = test_data['v_rel']
# amp = test_data['amp']
# plt.plot(v_rel,amp)
# plt.show()

# rs, vs = rot_array(d_struct)
# print(rs)
# print(vs)
# plt.plot(rs, vs,'.')
# plt.show()