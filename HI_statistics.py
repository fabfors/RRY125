from astropy.coordinates.sky_coordinate import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import astropy
from astropy import coordinates as coords
from astropy import units 
from astropy.units import cds
# import cartopy.crs as ccrs



import scipy as scp

import typing as typ
import os
from os.path import isfile, join
import json

# data_path = "data/"

# H0 = 72.1 # (km/s)/Mpc
# files = [join(data_path,f) for f in os.listdir(data_path) if isfile(join(data_path,f))]


def coordinates(fs):
    contents_file_path = "file_contents_generated.txt"
    g_coords = {}
    d_struct = data_structure(fs)
    for f in fs:
        to_rad = lambda x: float(x) * np.pi / 180.0
        
        
        
        
        print(aa_coord.ra)
        print(aa_coord.ra.unit)

        g_coords[f] = {}
        g_coords[f]["galactic"] = g_coord
        g_coords[f]["j2000"] = aa_coord
        
    with open(contents_file_path,'w') as f:
        for key in list(g_coords):
            glat, glon = (lambda x: (x.l, x.b))(g_coords[key]["galactic"])
            jlat, jlon = (lambda x: (x.ra, x.dec))(g_coords[key]["j2000"])
            f.write(join(key,'\n',str(glat), " " , str(glon), '\n'))
            f.write(join(str(jlat), " " , str(jlon), '\n'))
    return 
    


# plot_galactic(files)
# plt.show()




def plot_file_data(f,save=False):

    fig = plt.figure() #, axs = plt.subplots(1,len(f))

    d_struct = data_structure(f,len(f))
    
    for i, key in enumerate(list(d_struct)):
        glon = d_struct[key]['l']
        glat = d_struct[key]['b']
        x = d_struct[key]['v_rel']
        y = d_struct[key]['amp']

        
        plt.plot(x,y)
        
        plt.gca().set(title  = join(str(glon),  str(glat)), \
            xlabel = "Velocity relative to LSR [km/s]", \
            ylabel = "Uncalibrated antenna temperature [K]")
        
        # if save:
        fig.tight_layout(h_pad=2) 
        # print(key)
        key_name = key.split('/')[1].split('.')[0]
        key_name = join(key_name,'.png').replace('/','')
        if not isfile(join("images/", key_name)):
            plt.savefig(join("images/", key_name ) )
        
        
        plt.cla()

        
        # axs.plot(x,y)
    
    return fig

# fig = plot_file_data(files,True)
#   data_structure : FilePath -> (Date,GLON,GLAT,Array)
def data_structure(data_path, save_json=False, json_name=None, new_files=False, new_target_path=None):
    file_list = list(map(lambda x: join(data_path,x),os.listdir(data_path)))
    data_dict = {}
    save_counter = 0
    N = len(file_list)
    for file_path in file_list:
        print(file_path.split('.')[1])
        if not file_path.split('.')[1] == "txt": continue
        print(f"d_structure: {file_path}")
        with open(file_path) as fil:

            data = fil.readlines()
            data_A = np.genfromtxt(file_path)
            # print(file_path)
            # print(data_A)
            
            header = data[0:8]
            timing = (header[2].split("=")[1]).split("T")
            
            
            l = float((header[4].split("=")[1]).split('\n')[0])
            b = float((header[5].split("=")[1]).split('\n')[0])
            
            # g_coord = coords.SkyCoord(l=l,b=b,frame=coords.Galactic,unit=cds.deg)
            date = timing[0]
            time = timing[1]
            v_rel = data_A[:,0]
            amp = data_A[:,1]

            data_dict[file_path] = {}
            data_dict[file_path]["date"] = date
            data_dict[file_path]["time"] = time
            data_dict[file_path]["l"] = l
            data_dict[file_path]["b"] = b
            data_dict[file_path]["v_rel"] = v_rel.tolist()
            data_dict[file_path]["amp"] = amp.tolist()
            data_dict[file_path]['file_index'] = save_counter
            
            if l >= 0 and l < 90:
                data_dict[file_path]['quadrant'] = 1
            if l >= 90 and l < 180:
                data_dict[file_path]['quadrant'] = 2
            if l >= 180 and l < 270:
                data_dict[file_path]['quadrant'] = 3
            if l >= 270 and l < 360:
                data_dict[file_path]['quadrant'] = 4

            if new_files:
                datum = data_dict[file_path]
                new_file_name = f"Q{datum['quadrant']}_{str(l)[:5]}_{str(b)[:5]}_data_{save_counter}.txt"
                np.savetxt(os.path.join(new_target_path,new_file_name),data_A)
                print(f"Saved new data file {save_counter}/{N}")
            save_counter += 1
    if save_json:
        with open(json_name, 'w') as fp:
            json.dump(data_dict,fp)
        print("data dict dumped")
    return data_dict

def rel_freq(rel_v):
    c = 3000000000 # m/s
    f0 = 1420.4
    return -rel_v * (c - rel_v) / c

def r_dist(rel_v):
    # rel_v : km/s
    return [rel_v[i] * (1 / H0) for i in range(len(rel_v))] # : Mpc





def suc(n: int) -> int:
    return n + 1
    
    
def galactic_ax_w_image(x : typ.List[float], y : typ.List[float]) -> plt.Axes:
    img = mpimg.imread('/home/fabfors/Pictures/gaia-milky-way.jpg')
    # imgplot = plt.imshow(img)


    eq = coords.SkyCoord(x, y, frame='galactic', unit=units.deg)
    gal = eq.galactic
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="mollweide")
    plt.grid(True)
    ax.imshow(img, transform=ccrs.Mollweide())
    ax.scatter(gal.l.wrap_at('180d').radian, gal.b.radian)

    return ax

def galactic_ax(x : typ.List[float], y : typ.List[float]) -> plt.Axes:
    # gal = f(x,y)
    eq = coords.SkyCoord(x, y, frame='galactic', unit=units.deg)
    gal = eq.galactic
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="mollweide")
    plt.grid(True)
    ax.scatter(gal.l.wrap_at('180d').radian, gal.b.radian)

    return ax


# d_struct = data_structure(files,save=True)
# keys = list(d_struct)

# ls = []
# bs = []
# for key in keys:
#     ls.append(d_struct[key]["l"])
#     bs.append(d_struct[key]["b"])
#     if d_struct[key]['l'] > 180 - 2 and d_struct[key]['l'] < 180 + 2:
#         print(key)

# print(f(ls,bs))
# print(len(files))

# print(keys[np.argmax(ls)])
# ax = galactic_ax(ls, bs)
# ax = galactic_ax_w_image(ls,bs)
# plt.show()
# plot_file_data(files,save=True)
# print(f(ras,decs))
# plt.show()

# fil = np.genfromtxt("data/spectrum_47781.txt")
# fil_T = fil.T
# plt.plot(fil_T[0],fil_T[1])
# plt.show()

# with open("data_struct.json","w") as outfile:
#     json.dump(d_struct, outfile)
