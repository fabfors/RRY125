from astropy.coordinates.sky_coordinate import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.defchararray import find
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
import math
import shutil

data_path = 'data/'
# data_path_milos = 'data/milos/'


# file_list_milos = list(map(lambda x: join(data_path_milos, x), os.listdir(data_path_milos)))
# file_list = file_list + file_list_milos
# rand_n = np.random.randint(0,len(file_list),16)

# data_list = [np.asarray(pd.read_csv(join(data_path,file_list[rand_n[i]]),sep=" ", header=None,skiprows=8))\
#     for i in range(len(rand_n))]

# data_list = [np.asarray(pd.read_csv(join(data_path,file_list[i]),sep=" ", header=None,skiprows=8)) for i in range(len(file_list))]
# data_list_milos = [np.asarray(pd.read_csv(join(data_path_milos,file_list_milos[i]),sep=" ", header=None,skiprows=8)) for i in range(len(file_list_milos))]

# data = pd.read_csv('data/spectrum_47593.txt', sep=" ", header=None,skiprows=8)
# data = np.asarray(data)

# Generate number of peaks

# Generate peak heights

# Generate guesses
# print(data)

def guesses(data):
    peak_ixs, y = signal.find_peaks(data[:,1],height=20,prominence=10)
    peak_heights = y['peak_heights']
    
    ws = [10 for i in range(len(peak_ixs))]

    outlist = np.concatenate(np.c_[1/peak_heights, peak_heights, ws])
    return outlist, len(peak_heights) 


def last_gaussian(data):
    guesss, n = guesses(data)
    params, success = mcfit.opt(np.stack(guesss), data, n)
    N = len(params)
    return mcfit.gaussian(data[:,0],params[N-3],params[N-2],params[N-1],0)

def save_plt_fig(datum, peak_ixs):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    v_rel = np.flip(datum[:,0])
    amp = datum[:,1]
    amp_filtered = signal.wiener(amp)
    amp_min = np.abs(np.min(amp_filtered))
    amp_max = np.abs(np.max(amp_filtered))

    amp_filtered = amp_min + amp_filtered
    ax.plot(v_rel,amp_filtered)
    for i, ix in enumerate(peak_ixs):
        ax.plot(v_rel[ix], amp_min + amp[ix], 'r.')
    fig.savefig(f"peak_images/data_index_{image_counter}.png")
    plt.close(fig) 
    ax.cla()

def save_plt_figs(data_list):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    image_counter = 0
    for i, datum in enumerate(data_list):
        v_rel = np.flip(datum[:,0])
        amp = datum[:,1]
        amp_filtered = signal.wiener(amp)
        amp_min = np.abs(np.min(amp_filtered))
        amp_max = np.abs(np.max(amp_filtered))

        amp_filtered = amp_min + amp_filtered
        ax.plot(v_rel,amp_filtered)
        peak_ixs, peak_data = signal.find_peaks(amp_filtered,height=(10,),prominence=amp_max/20,width=(1,50))
        for i, ix in enumerate(peak_ixs):
            ax.plot(v_rel[ix], amp_min + amp[ix], 'r.')
        fig.savefig(f"peak_images/data_index_{image_counter}.png")
        image_counter += 1
        plt.close(fig) 
        ax.cla()

def plot_peaks(datum):
    v_rel = np.flip(datum['v_rel'])
    amp = np.flip(datum['amp'])
    amp_filtered = signal.wiener(amp)
    amp_min = np.abs(np.min(amp_filtered))
    amp_max = np.abs(np.max(amp_filtered))
    amp_filtered = amp_min + amp_filtered
    peak_ixs, peak_data = signal.find_peaks(amp_filtered,height=(10,),prominence=amp_max/20,width=(1,50))


def pre_process(v_rel, amp, window_w):
    v_rel_f = np.flip(v_rel)
    amp_f = np.flip(amp)
    amp_filtered = signal.wiener(amp_f, window_w)
    amp_min = np.abs(np.min(amp_filtered))
    amp_filtered -= amp_min
    return v_rel_f, amp_filtered

def plot_gaussian_fits(d_struct, guesses, separate=True, quadrantI=True, plot=True):
    params = []
    n = len(guesses)
    q_list = {}
    if quadrantI:
        l_c = 1
        b_c = 3
        for key in list(d_struct):
            l = d_struct[key]['l']
            b = d_struct[key]['b']
            if b < 0 + b_c and b > 0 - b_c: # and l <= 90 + l_c and l >= 0 - l_c:
                q_list[key] = d_struct[key]
    else:
        q_list = d_struct
    if plot:
        axs_ixs = [(i,j) for i in range(len(q_list) // 4) for j in range(4)]
        fig, axs = plt.subplots(len(q_list) // 4, 4)
    
    for i, key in enumerate(list(q_list)):
        datum = q_list[key]
        v_rel = np.flip(datum['v_rel'])
        amp = np.flip(datum['amp'])

        amp_filtered = signal.wiener(amp,15)
        amp_min = np.min(amp_filtered)
        if amp_min < 0:
            amp_filtered = amp_filtered + np.abs(amp_min)
        elif amp_min > 0:
            amp_filtered = amp_filtered - amp_min

        data_filtered = np.c_[v_rel, amp_filtered]
        optim, success = mcfit.opt(guesses, data_filtered, n)
        params.append(optim)

        func = mcfit.n_gaussians(v_rel, optim, n)
        if plot:
            a = axs_ixs[i]
            ax = axs[a[0],a[1]]
            ax.plot(v_rel,amp_filtered)
            if separate:
                for param_i in range(0,len(optim),3):
                    ax.plot(v_rel, mcfit.gaussian(v_rel, optim[param_i], optim[param_i + 1], optim[param_i + 2], offset=0))
            else:
                ax.plot(v_rel, func)
    if plot:
        fig.tight_layout()
        plt.show()
    
    return params

def plot_n_peaks(d_struct, window_w, n=16, plot=True, save=False, quadrantI=True):
    peaks = []
    if plot:
        axs_ixs = [(i,j) for i in range(n // 4) for j in range(4)]
        fig, axs = plt.subplots(n // 4,4)
    if save:
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        image_counter = 0
    q_list = {}

    if quadrantI:
        l_c = 1
        b_c = 3
        for key in list(d_struct):
            l = d_struct[key]['l']
            b = d_struct[key]['b']
            if b < 0 + b_c and b > 0 - b_c: # and l <= 90 + l_c and l >= 0 - l_c:
                q_list[key] = d_struct[key]
    else:
        q_list = d_struct
    for i, key in enumerate(list(q_list)):
        if i >= n: break
        glon = q_list[key]['l']
        glat = q_list[key]['b']
        v_rel = q_list[key]['v_rel']
        amp = q_list[key]['amp']
        v_rel, amp = pre_process(v_rel, amp, window_w )
        amp_max = np.max(amp)
        peak_ixs, peak_data = signal.find_peaks(amp,height=(10,),prominence=5,width=(1,50))
        if plot:
            a = axs_ixs[i]
            axis = axs[a[0],a[1]]
            axis.plot(v_rel,amp)
            axis.set(title  = join(str(glon),  str(glat)), \
            xlabel = "Velocity relative to LSR [km/s]", \
            ylabel = "Uncalibrated antenna temperature [K]")
            for i, ix in enumerate(peak_ixs):
                axis.plot(v_rel[ix], amp[ix], 'r.')
        peaks.append(v_rel[peak_ixs[-1]])
        
        if save:
            ax.plot(v_rel, amp)
            ax.set(title  = join(str(glon),  str(glat)), \
            xlabel = "Velocity relative to LSR [km/s]", \
            ylabel = "Uncalibrated antenna temperature [K]")
            for i, ix in enumerate(peak_ixs):
                ax.plot(v_rel[ix], amp[ix], 'r.')
            fig.savefig(f"quadrant_I/{str(glon)[:5]}_data_index_{image_counter}.png")
            image_counter += 1
            plt.close(fig) 
            ax.cla()    
    if plot:
        fig.tight_layout()
        plt.show()

    return peaks

def plot_rcurve(data):
    V = data[:,2]
    R = data[:,3]
    sigma = data[:,4]
    plt.errorbar(R, V, fmt='b.', yerr=sigma,ecolor='red')
    plt.xlabel("Distance from galactic centre [kpc]")
    plt.ylabel("Rotational velocity")
    plt.show()

def right_gaussians(d_struct):
    guesses = (15 - 10) * np.random.random_sample(6) + 10
    params = plot_gaussian_fits(d_struct, guesses, separate=True, quadrantI=True, plot=False)
    three_params = params
    # for param_list in params:
    #     means = [param_list[i] for i in range(1,len(param_list), 3)]
    #     mean_max = np.max(means)
    #     mm_ix = np.where(param_list == mean_max)[0]
    #     # print(np.where(param_list == mean_max)[0])
    #     mean_max = param_list[mm_ix]
    #     height = param_list[mm_ix - 1]
    #     width = param_list[mm_ix + 1]
    #     three_params.append([height[0], mean_max[0], width[0]])
    q_list = {}
    l_c = 1
    b_c = 3
    for key in list(d_struct):
        l = d_struct[key]['l']
        b = d_struct[key]['b']

        if b < 0 + b_c and b > 0 - b_c: # and l <= 90 + l_c and l >= 0 - l_c:
            q_list[key] = d_struct[key]
    three_params = np.asarray(three_params)
    qI_v_rels = three_params[:,1]
    qI_sigmas = three_params[:,2]
    lbv_rels = []
    for i, key in enumerate(list(q_list)):
        l = math.radians(q_list[key]['l'])
        b = math.radians(q_list[key]['b'])
        v_rel = qI_v_rels[i]
        V = v_rel + V0 * np.sin(l)
        R = R0 * np.sin(l)
        sigma = qI_sigmas[i]
        lbv_rels.append([l, b, V, R, sigma])
    lbv_rels = np.asarray(lbv_rels)
    return lbv_rels

def struct_peaks(d_struct, plot=False, save=False, folder=None, n=None):
    keys = list(d_struct)
    image_counter = 0
    # print(keys)
    if plot:
        axs_ixs = [(i,j) for i in range(n // 4) for j in range(4)]
        fig_1, axs = plt.subplots(n // 4,4)

    for i, key in enumerate(keys):
        if n:
            if i >= n:
                break
        datum = d_struct[key]
        v_rel = np.flip(datum['v_rel'])
        amp = np.flip(np.asarray(datum['amp']))

        amp_filtered = signal.wiener(amp, 10)
        amp_min = np.min(amp_filtered)
        # amp_max = np.max(amp_filtered)
        if amp_min < 0:
            amp_filtered = amp_filtered - amp_min
        elif amp_min > 0:
            amp_filtered = amp_filtered - amp_min
        d_struct[key]['amp'] = amp_filtered.tolist()
        d_struct[key]['v_rel'] = v_rel.tolist()
        peak_ixs, _ = signal.find_peaks(amp_filtered,height=20,prominence=0.1, wlen=50, distance=10)
        d_struct[key]['peak_ixs'] = peak_ixs.tolist()
        d_struct[key]['l'] = math.radians(datum['l'])
        d_struct[key]['b'] = math.radians(datum['b'])
        d_struct[key]['peak_vs'] = []
        d_struct[key]['peak_amps'] = []
        for v_ix in peak_ixs:
            d_struct[key]['peak_vs'].append(v_rel[v_ix])
            d_struct[key]['peak_amps'].append(amp[v_ix])

        if plot:
            glon = datum['l']
            glat = datum['b']
            axi = axs_ixs[i][0]
            axj = axs_ixs[i][1]
            axs[axi,axj].plot(v_rel, amp_filtered)
            # t = ("l: TEST TEXT")
            # axs[axi,axj].text(5, 20,t,wrap=True)
            for i, ix in enumerate(peak_ixs):
                axs[axi,axj].scatter(v_rel[ix], amp_filtered[ix], c='red')
            axs[axi,axj].set(title=join(str(glon),  str(glat)), \
                xlabel = "Velocity relative to LSR [km/s]", \
                ylabel = "Uncalibrated antenna temperature [K]")

        if save:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(v_rel, amp_filtered)
            glon = math.degrees(datum['l'])
            glat = math.degrees(datum['b'])
            ax.set(title=join(str(glon)[:7],  str(glat)[:7]), \
            xlabel = "Velocity relative to LSR [km/s]", \
            ylabel = "Uncalibrated antenna temperature [K]")
            for i, ix in enumerate(peak_ixs):
                ax.plot(v_rel[ix], amp_filtered[ix], 'r.')
            ## TODO: Send images from different quadrants to separate folders
            fn = f"{folder}Q{str(datum['quadrant'])[:1]}_{str(glon)[:5]}_{str(glat)[:5]}_data_index_{image_counter}.png"
            # with open(fn) as fp:
            fig.savefig(fn)
            print(f"image: {image_counter}")
            # image_counter += 1
            plt.close(fig) 
            ax.cla()  
        glon = math.degrees(datum['l'])
        glat = math.degrees(datum['b'])
        # print(f"quadrant_I/Q{str(datum['quadrant'])[:1]}_{str(glon)[:5]}_{str(glat)[:5]}_data_index_{image_counter}.png")
        image_counter += 1
    
    return d_struct

def fix_gal_pos(q_list):
    for key in list(q_list):
        datum = q_list[key]
        l = datum['l']
        b = datum['b']
        v_rel_peaks = q_list[key]['peak_vs']
        if type(v_rel_peaks) is np.float64:
            v_rel_peaks = np.asarray([v_rel_peaks])
        for v_peak in v_rel_peaks:
            if b > math.radians(2) or b < math.radians(-2):
                q_list[key]['r'] = None
                continue
            R = R0 * V0  * np.sin(l) / (V0 * np.sin(l) + v_peak)
            r_pos = np.sqrt(R**2 - R0**2 * np.sin(l)**2) + R0 * np.cos(l)
            r_neg = -np.sqrt(R**2 + R0**2 * np.sin(l)**2) + R0 * np.cos(l)

            if q_list[key]['quadrant'] == 1.:
                print(f"Q{q_list[key]['quadrant']} r_pos:{r_pos} r_neg:{r_neg}")
                # Either is nan
                if str(r_pos) == "nan" or str(r_neg) == "nan":
                    q_list[key]['r'] = None
                # Both positive
                elif r_pos > 0 and r_neg > 0:
                    q_list[key]['r'] = [r_pos, r_neg]
                # only r_pos positive 
                elif r_pos > 0 and r_neg < 0:
                    q_list[key]['r'] = r_pos
                else:
                    header = f"Q{q_list[key]['quadrant']} {str(math.degrees(l))[:5]} {str(math.degrees(b))[:5]} {q_list[key]['file_index']} {key}"
                    x_pos = r_pos * np.cos(l - np.pi / 2)
                    y_pos = r_pos * np.sin(l - np.pi / 2)
                    x_neg = r_neg * np.cos(l - np.pi / 2)
                    y_neg = r_neg * np.sin(l - np.pi / 2)
                    
                    print(header)
                    print(v_rel_peaks)
                    print(f"Vr: {v_peak}")
                    print(f"R: {R}")
                    print(f"r_pos: {r_pos}")
                    print(f"r_neg: {r_neg}")
                    print(f"x_pos = {x_pos}, y_pos = {y_pos}")
                    print(f"x_neg = {x_neg}, y_neg = {y_neg}")
                    msg = f"Write 1 for pos, 2 for neg, 3 for None"
                    print(msg)
                    inp = input()
                    if inp == "1":
                        print("yup read 1")
                        q_list[key]['r'] = r_pos
                    elif inp == "2":
                        q_list[key]['r'] = r_neg
                    elif inp == "3":
                        q_list[key]['r'] = None
                    else: print("Something went wrong")


            if q_list[key]['quadrant'] == 2.:
                print(f"Q{q_list[key]['quadrant']} r_pos:{r_pos} r_neg:{r_neg}")
                if str(r_pos) == 'nan' and str(r_neg) == 'nan': print('Q2 both nan wt\nf')
                elif str(r_pos) == 'nan':
                    if not str(r_neg) == 'nan' and r_neg > 0:
                        q_list[key]['r'] = r_neg
                    else: q_list[key]['r'] = None
                elif str(r_neg) == 'nan':
                    if not str(r_pos) == 'nan' and r_pos > 0:
                        q_list[key]['r'] = r_pos
                    else: q_list[key]['r'] = None
                elif not str(r_pos) == 'nan' and not str(r_neg) == 'nan':
                    if r_pos > 0 and r_neg > 0:
                        q_list[key]['r'] = [r_pos, r_neg]
                    if r_pos > 0:
                        q_list[key]['r'] = r_pos
                    elif r_neg > 0:
                        q_list[key]['r'] = r_neg
                    else: q_list[key]['r'] = None
                else: print("Q2: Something fucked up\n")

            if q_list[key]['quadrant'] == 3.:
                print(f"Q{q_list[key]['quadrant']} r_pos:{r_pos} r_neg:{r_neg}")
                if str(r_pos) == 'nan' and str(r_neg) == 'nan': print('Q3 both nan wtf')
                elif str(r_pos) == 'nan':
                    if not str(r_neg) == 'nan' and r_neg > 0:
                        q_list[key]['r'] = r_neg
                    else: q_list[key]['r'] = None
                elif str(r_neg) == 'nan':
                    if not str(r_pos) == 'nan' and r_pos > 0:
                        q_list[key]['r'] = r_pos
                    else: q_list[key]['r'] = None
                elif not str(r_pos) == 'nan' and not str(r_neg) == 'nan':
                    if r_pos > 0 and r_neg > 0:
                        q_list[key]['r'] = [r_pos, r_neg]
                    if r_pos > 0:
                        q_list[key]['r'] = r_pos
                    elif r_neg > 0:
                        q_list[key]['r'] = r_neg
                    else: q_list[key]['r'] = None
                else: print("Q3: Something fucked up\n")

            if q_list[key]['quadrant'] == 4.:
                print(f"Q{q_list[key]['quadrant']} r_pos:{r_pos} r_neg:{r_neg}")
                q_list[key]['r'] = None
                print(f"Q4: pos: {str(r_pos)[:5]}. neg: {str(r_neg)[:5]}\n")

            ## -- Calculate x, y coordinates ---------------------------------
            if q_list[key]['r']:
                rs = q_list[key]['r']
                l = q_list[key]['l']
                b = q_list[key]['b']
                v_rels = v_rel_peaks
                if type(rs) is np.float64:
                    xs = rs * np.cos(l - np.pi / 2)
                    ys = rs * np.sin(l - np.pi / 2)
                    q_list[key]['xs'] = xs
                    q_list[key]['ys'] = R0 + ys
                else:
                    xs = np.zeros_like(rs)
                    ys = np.zeros_like(rs)
                    for i in range(len(rs)):
                        xs[i] = rs[i] * np.cos(l - np.pi / 2)
                        ys[i] = rs[i] * np.sin(l - np.pi / 2)
                        q_list[key]['xs'] = xs.tolist()
                        q_list[key]['ys'] = (R0 + ys).tolist()
            else:
                q_list[key]['xs'] = None
                q_list[key]['ys'] = None
                q_list[key]['zs'] = None

    return q_list

def gal_position(q_list,ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot()
    colors = ['cyan', 'magenta', 'green', 'red']
    plot_n = 0
    for key in q_list.keys():
        # print(key)
        if 'xs' in q_list[key] and 'ys' in q_list[key]:
            # print(f"Q{q_list[key]['quadrant']}: xs found {q_list[key]['xs']}")
            if q_list[key]['xs'] and q_list[key]['ys']:
                xs = q_list[key]['xs']
                # print(f"if q_list[key]['xs'] = {q_list[key]['xs']}")
                if type(xs) is np.float64:
                    xs = np.asarray([xs])
                    # print(f"len(xs)=1")
                else:
                    xs = np.asarray(xs)
                    # print(f"len(xs)={len(xs)}")
                ys = np.asarray(q_list[key]['ys'])
                q = q_list[key]['quadrant']
                
                print(f"Q{q_list[key]['quadrant']}")
                print(f"xs: {xs}")
                print(f"ys: {ys}")
                if len(xs) > 1:
                    ax.scatter(xs, ys,c=colors[int(q)-1])
                    # print(f"plotted {str(xs)} {str(ys)}")
                    plot_n += len(xs)
                else:
                    ax.scatter(xs, ys, color=colors[int(q)-1])
                    # print(f"plotted {str(xs)[:5]} {str(ys)[:5]}")
                    plot_n += len(xs)
                
        # else: 
        #     print(f"fuck: {key}\nl: {math.degrees(q_list[key]['l'])}\nb: {math.degrees(q_list[key]['b'])}")
    print(f"Plotted {plot_n} points")
        
    ax.scatter(0,0,marker='s',c='black')
    ax.scatter(0,R0,marker='o',c='black')
    ax.set(xlabel="x (kpc)", ylabel="y (kpc)", xlim=(-15,15), ylim=(-20,20))
    ax.grid()
    return ax

def gal_position_1(lbv_rels):
    ls = lbv_rels[:,0]
    Vr = lbv_rels[:,2]
    qs = lbv_rels[:,5]
    Rs = np.zeros_like(ls)
    xs = np.zeros_like(ls)
    ys = np.zeros_like(ls)
    
    for i in range(len(Rs)):
        Rs[i] = R0 * V0 * np.sin(ls[i]) / (V0 * np.sin(ls[i]) + Vr[i])
        # print(Rs[i]**2, R0**2, np.sin(ls[i])**2, Rs[i]**2 - R0**2 * np.sin(ls[i])**2)
        rs = Rs[i]**2 - R0**2 * np.sin(ls[i])**2
        if rs < 0:
            print("rs < 0 ", qs[i])
            rs = np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i])
        else:
            print("rs > 0 ", qs[i])
            rs = np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i])
            
        

        xs[i] = rs * np.cos(ls[i] - np.pi / 2)
        ys[i] = rs * np.sin(ls[i] - np.pi / 2)
    
    out_arr = np.c_[ls,xs,ys,qs]

    print("gal_pos.shape: ", out_arr.shape)
    return out_arr

def gal_position_3d(q_list):
    ls = lbv_rels[:,0]
    bs = lbv_rels[:,1]
    Vr = lbv_rels[:,2]
    qs = lbv_rels[:,5]
    Rs = np.zeros_like(ls)
    xs = np.zeros_like(ls)
    ys = np.zeros_like(ls)
    zs = np.zeros_like(ls)
    
    for i in range(len(Rs)):
        # print(Vr[i])
        Rs[i] = R0 * V0 * np.sin(ls[i]) / (V0 * np.sin(ls[i]) + Vr[i])
        # print(Rs[i]**2, R0**2, np.sin(ls[i])**2, Rs[i]**2 - R0**2 * np.sin(ls[i])**2)
        rs = Rs[i]**2 - R0**2 * np.sin(ls[i])**2
        print(f"Q{qs[i]}: ", str(Rs[i])[:4], R0)
        print(f"Vr: {Vr[i]}")
        print(f"l: {ls[i] * 180 / np.pi}")
        print(f"b: {bs[i] * 180 / np.pi}")

        # print(rs > 0, qs[i])
        # if qs[i] == 1.:
        #     print("Q I: ", 
        #         np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]),
        #         np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]))
        # elif qs[i] == 4.:
        #     print("Q IV: ", 
        #         np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]),
        #         np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]))
        # elif qs[i] == 2.:
        #     print("Q II: ", 
        #         np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]),
        #         np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]))
        # elif qs[i] == 3.:
        #     print("Q III: ", 
        #         np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]),
        #         np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i]))

        if rs < 0:
            # print("rs < 0 ", qs[i])
            rs = np.sqrt(-Rs[i]**2 + R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i])
        else:
            # print("rs > 0 ", qs[i])
            rs = np.sqrt(Rs[i]**2 - R0**2 * np.sin(ls[i])**2) + R0 * np.cos(ls[i])
            
        

        xs[i] = rs * np.cos(ls[i] - np.pi / 2) * np.sin(np.pi / 2 - bs[i])
        ys[i] = rs * np.sin(ls[i] - np.pi / 2) * np.sin(np.pi / 2 - bs[i])
        zs[i] = rs * np.cos(np.pi / 2-bs[i])
        # xs[i] = rs * np.cos(ls[i] - np.pi / 2) * np.sin(bs[i])
        # ys[i] = rs * np.sin(ls[i] - np.pi / 2) * np.sin(bs[i])
        # zs[i] = rs * np.cos(bs[i])    
    # print(Rs**2 - R0**2 * np.sin(ls)**2)
    # rs_pos = np.sqrt(Rs**2 - R0**2 * np.sin(ls)**2) + R0 * np.cos(ls)
    # rs_neg = -np.sqrt(Rs**2 - R0**2 * np.sin(ls)**2) + R0 * np.cos(ls)
    # xs = []
    # ys = []
    # for i, q in enumerate(qs):
    #     if q == 2 or q == 3:
    #         xs.append(rs_pos[i] * np.cos(ls[i] - np.pi / 2))
    #         ys.append(rs_pos[i] * np.sin(ls[i] - np.pi / 2))
    #     else:
    #         xs.append(rs_pos[i] * np.cos(ls[i] - np.pi / 2))
    #         ys.append(rs_pos[i] * np.sin(ls[i] - np.pi / 2))

    out_arr = np.c_[ls,bs,xs,ys,zs,list(map(int,qs))]
    with open("pos_data.txt",'w') as f:
        np.savetxt(f,out_arr,delimiter=",")
    

    print("gal_pos.shape: ", out_arr.shape)
    return out_arr

def move_files(files,source_dir,target_dir):
    for file_name in files:
        dir_file_split = file_name.split('/')
        if  dir_file_split[0] == 'data':
            shutil.move(file_name, os.path.join(target_dir,dir_file_split[1]))
        else:
            return print("fix this")

def amend_dict(file_path):
    with open(file_path) as fp:
        q_list = json.load(fp)
        print(list(q_list[list(q_list)[0]]))
V0 = 220 # km/s
R0 = 8.5 # kpc
def main():
    
    # with open("quadrant_I/HI_data.json") as fp:
    #     q_list = json.load(fp)
    #     fix_gal_pos(q_list)
        # struct_peaks(q_list,save=True,folder="quadrant_I/milos_images/")
    qs = []
    ls = []
    bs = []
    vs = []
    # with open("quadrant_I/HI_data.json") as fp:
    #     q_list = json.load(fp)

    #     qq_list = fix_gal_pos(q_list)
    #     gal_position(qq_list)

    d_struct = HI.data_structure('data/', new_files=False, new_target_path=None)
    q_list = struct_peaks(d_struct, plot=False,n=None, save=False)
    qq_list = fix_gal_pos(q_list)
    
    dm_struct = HI.data_structure('milos/', new_files=False, new_target_path=None)
    qm_list = struct_peaks(dm_struct, plot=False,n=None, save=False)
    qqm_list = fix_gal_pos(qm_list)
    ax = gal_position(qq_list)
    gal_position(qqm_list,ax)
    plt.show()
    # dump_dict(q_list,"quadrant_I/HI_data.json")

    # d_struct = HI.data_structure('data/', new_files=False, new_target_path=None)
    # q_list = struct_peaks(d_struct, plot=False,n=None, save=False)
    
    # dump_dict(qq_list,"quadrant_I/HI_data.json")
        
    
        # for key in d_struct.keys():
        #     l = d_struct[key]['l']
        #     b = d_struct[key]['b']
        #     if d_struct[key]['quadrant'] == 1. and b > -1 and b < 1:
        #         v_rel_peaks = d_struct[key]['peak_vs']
        #         if type(v_rel_peaks) is np.float64:
        #             v_rel_peaks = np.asarray([v_rel_peaks])
        #         header = f"Q{d_struct[key]['quadrant']} {str(math.degrees(l))[:5]} {str(math.degrees(b))[:5]} {d_struct[key]['file_index']} {key}"
        #         print(header)
        #         for v_peak in v_rel_peaks:
        #             R = R0 * V0  * np.sin(l) / (V0 * np.sin(l) + v_peak)
        #             r_pos = np.sqrt(R**2 - R0**2 * np.sin(l)**2) + R0 * np.cos(l)
        #             r_neg = -np.sqrt(R**2 + R0**2 * np.sin(l)**2) + R0 * np.cos(l)
        #             print(f"Vr: {v_peak}")
        #             print(f"R: {R}")
        #             print(f"r_pos: {r_pos}")
        #             print(f"r_neg: {r_neg}")
        #         print()
    
    # d_struct = HI.data_structure('data/', new_files=False, new_target_path=None)
    # q_list = struct_peaks(d_struct, plot=False,n=None, save=False)
    # dump_dict(q_list,"quadrant_I/HI_data.json")
    
    
    
    
    # dict_path = 'quadrant_I/HI_data.json'
    # amend_dict(dict_path)

    # d_struct = HI.data_structure('data/', new_files=False, new_target_path=None)
    # q_list = struct_peaks(d_struct, plot=False,n=None, save=True)
    
def dump_dict(d_struct, json_path):
    # json_path = "quadrant_I/HI_data.json"
    keys = list(d_struct)
    # for d_key in d_struct[keys[0]].keys():
    #     print(d_key)
    #     print(type(d_struct[keys[0]][d_key]))
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path,'x') as fp:
        json.dump(d_struct, fp)
        print("dump_dict dumped")
    
    # q_list = fix_gal_pos(q_list)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # colors = ['cyan', 'magenta', 'green', 'black']
    # for key in list(q_list):
    #     if q_list[key]['xs']:
    #         xs = q_list[key]['xs']
    #         ys = q_list[key]['ys']
    #         zs = q_list[key]['zs']
    #         q = q_list[key]['quadrant']
            
    #         for i in range(len(xs)):
    #             ax.scatter(xs[i], R0 + ys[i],zs[i],c=colors[int(q)])
    # ax.scatter(0,0,0,marker='s',c='black')
    # ax.scatter(0,R0,0,marker='o',c='black')
    # ax.set(xlabel="x (kpc)", ylabel="y (kpc)", xlim=(-15,15), ylim=(-15,15), zlim=(-20,20))
    # ax.grid()
    # plt.show()
    


if __name__ == "__main__":
    main()