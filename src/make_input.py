"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Fev 26, 2023
"""

import os
import sys
import datetime
import csv
import numpy as np
from scipy.interpolate import interp1d
from pickle import dump

from folders import path_Vr_data, path_piezo_data, path_input

sys.path.append('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Python_modules/')
from ndimcube.ndimcube import NDimCube


import warnings
warnings.filterwarnings("ignore")





### FUNCTIONS -------------------------------------------------------------------------------------
def resamp(f, v, wavelengths):
    w = v / f
    func_v = interp1d(w, v, fill_value='extrapolate')
    v_resamp = func_v(wavelengths)
    return wavelengths, v_resamp


def extract_vr(freqs, N_samples, date_start, date_end, point, Nw, wavelengths, FolderPath):
    N_freqs = len(freqs)
    vr_db = np.full((N_samples, N_freqs), np.nan)
    for f_i, freq in enumerate(freqs):
        tmp = np.load(f"{FolderPath}{date_start.strftime('%Y-%m-%d')}_{date_end.strftime('%Y-%m-%d')}/vr_{freq:.2f}Hz_point{point}_dateStart{date_start.strftime('%Y-%m-%d')}_dateEnd{date_end.strftime('%Y-%m-%d')}_withNans.npy")
        for s_i in range(Nw//2, N_samples-Nw//2):
            vr_db[s_i, f_i] = np.nanmean(tmp[s_i-Nw//2 : s_i+Nw//2+1])
    vr_db_tmp = np.copy(vr_db)
    vr_db = np.full((N_samples, len(wavelengths)), np.nan)
    for s_i in range(Nw//2, N_samples-Nw//2):
        _, vr_db[s_i, :] = resamp(freqs, vr_db_tmp[s_i, :], wavelengths)
    return wavelengths, vr_db
### -----------------------------------------------------------------------------------------------





### DAYS ------------------------------------------------------------------------------------------
date_start = datetime.datetime.strptime('2022-12-30 00:00:00', '%Y-%m-%d %H:%M:%S')
date_end = datetime.datetime.strptime('2023-09-03 00:00:00', '%Y-%m-%d %H:%M:%S')
days = []
day = datetime.timedelta(days=1)
date = date_start
i = 0
while date <= date_end:
    days.append(date)
    date += day
    i += 1
days = np.array(days)
N_samples = len(days)

Nw = 0

print(f'\nFrom {date_start} to {date_end} -> {N_samples} days')
### -----------------------------------------------------------------------------------------------





### FREQS AND WAVELENGTHS -------------------------------------------------------------------------
freqs = [5.00, 5.21, 5.44, 5.67, 5.91, 6.16, 6.43, 6.70, 6.99, 7.29, 7.60, 7.92, 8.26, 8.62, 
             8.98, 9.37, 9.77, 10.19, 10.62, 11.08, 11.55, 12.04, 12.56, 13.10, 13.66, 14.24, 14.85, 
             15.48, 16.15, 16.84, 17.56, 18.31, 19.09, 19.91, 20.76, 21.64, 22.57, 23.53, 24.54, 25.59, 
             26.68, 27.82, 29.01, 30.25, 31.55, 32.90, 34.30, 35.77, 37.30, 38.89, 40.56, 42.29, 44.10, 
             45.98, 47.95, 50.00]
N_freqs = len(freqs)

wavelengths = np.arange(4, 15+0.5, 0.5)
N_wavelengths = len(wavelengths)

print(f"\nFrequencies : {N_freqs}")
print(f"Frequencies : {freqs}")

print(f"\nWavelengths : {N_wavelengths}")
print(f"Wavelengths : {wavelengths}")
### -----------------------------------------------------------------------------------------------




### ENTIRE Vr DATABASE ALONG t, x, y --------------------------------------------------------------
files = sorted(os.listdir(path_Vr_data))
files = [path_Vr_data+path for path in files]

file = files[0]
read_cube = NDimCube.load(file)
read_dim = read_cube.get_dimensions_scale()

xs = np.array(list(read_dim[0].values())[0])
ys = np.array(list(read_dim[1].values())[0])
fs = np.array(list(read_dim[2].values())[0])

xgrid, ygrid = read_cube.get_xy_local_grid()
xgrid_flat = xgrid.flatten()
ygrid_flat = ygrid.flatten()

N_points = len(xgrid_flat)
points = range(N_points)


print("\nBuilding vr(t,x,y,f) database")
db_vr_freq = np.full((len(days), len(xs), len(ys), N_freqs), np.nan)
for file_i, file in enumerate(files):
    read_cube = NDimCube.load(file)
    day = read_cube.time_stamp
    if day in days:
        day_i = np.where(days==read_cube.time_stamp)[0][0]
        for x_i, x in enumerate(xs):
            for y_i, y in enumerate(ys):
                for f_i, f in enumerate(fs):
                        db_vr_freq[day_i, x_i, y_i, f_i] = (read_cube.data[x_i, y_i, f_i])



print("\nBuilding vr(t,x,y,lbd) database")
db_vr_wlgt = np.full((len(days), len(xs), len(ys), N_wavelengths), np.nan)
w_resamp = None
for file_i, file in enumerate(files):
    read_cube = NDimCube.load(file)
    day = read_cube.time_stamp
    if day in days:
        day_i = np.where(days==read_cube.time_stamp)[0][0]
        for x_i, x in enumerate(xs):
            for y_i, y in enumerate(ys):
                        _, db_vr_wlgt[day_i, x_i, y_i, :] = resamp(freqs, db_vr_freq[day_i, x_i, y_i, :], wavelengths)

for day_i in range(len(days)):
    for x_i, x in enumerate(xs):
        for y_i, y in enumerate(ys):
            db_vr_wlgt[day_i, x_i, y_i, :] = db_vr_wlgt[day_i, x_i, y_i, :] / 2000
### -----------------------------------------------------------------------------------------------





### X PZ3 -----------------------------------------------------------------------------------------
points_PZ3 = [(63.00,04.75), (66.00,04.75), (69.00,04.75),
              (63.00,00.00), (66.00,00.00), (69.00,00.00),
             ]
X_PZ3 = None
for p_i, point in enumerate(points_PZ3):
    X_tmp = db_vr_wlgt[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0], :]

    if p_i == 0:
        X_PZ3 = np.copy(X_tmp)
    else :
        X_PZ3 = np.vstack((X_PZ3, X_tmp))
### -----------------------------------------------------------------------------------------------
        




### X PZ5 -----------------------------------------------------------------------------------------
points_PZ5 = [(93.00,19.00), (96.00,19.00), (99.00,19.00),
              (93.00,14.25), (96.00,14.25), (99.00,14.25),
              ]
X_PZ5 = None
for p_i, point in enumerate(points_PZ5):
    X_tmp = db_vr_wlgt[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0], :]   

    if p_i == 0:
        X_PZ5 = np.copy(X_tmp)
    else :
        X_PZ5 = np.vstack((X_PZ5, X_tmp))
### -----------------------------------------------------------------------------------------------





### y PZ3 -----------------------------------------------------------------------------------------
y_PZ3 = np.full((N_samples), np.nan)
file_path = f"{path_piezo_data}PZ3_interp300s.csv"
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] != '' and row[1] != '':
            date = datetime.datetime.strptime(f"{row[0]}", "%Y-%m-%d %H:%M:%S")
            if date >= date_start and date <= date_end and date.hour == 0 and date.minute == 0 and date.second == 0:
                i = np.where(days == date)[0][0]
                y_PZ3[i] = float(row[1])

np.save(f"{path_input}GWT_PZ3(t).npy", y_PZ3)

y_PZ3 = abs(y_PZ3)

y_tmp = np.copy(y_PZ3)
for i in range(len(points_PZ3)-1):
    y_PZ3 = np.hstack((y_PZ3, y_tmp))
### -----------------------------------------------------------------------------------------------





### y PZ5 -------------------------------------------------------------------------------------------
y_PZ5 = np.full((N_samples), np.nan)
file_path = f"{path_piezo_data}PZ5_interp300s.csv"
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] != '' and row[1] != '':
            date = datetime.datetime.strptime(f"{row[0]}", "%Y-%m-%d %H:%M:%S")
            if date >= date_start and date <= date_end and date.hour == 0 and date.minute == 0 and date.second == 0:
                i = np.where(days == date)[0][0]
                y_PZ5[i] = float(row[1])

np.save(f"{path_input}GWT_PZ5(t).npy", y_PZ5)

y_PZ5 = abs(y_PZ5)

y_tmp = np.copy(y_PZ5)
for i in range(len(points_PZ5)-1):
    y_PZ5 = np.hstack((y_PZ5, y_tmp))
### -----------------------------------------------------------------------------------------------





### REMOVING NaNs ---------------------------------------------------------------------------------
days_raw = np.copy(days)

print("\nBefore removing NaNs")
print(f"N_days:{len(days)}", f"N_wavelengths:{len(wavelengths)}")
print(f"N_points_PZ3:{len(points_PZ3)}", f"N_points_PZ5:{len(points_PZ5)}")
print(f"X_PZ3:{X_PZ3.shape}", f"y_PZ3:{y_PZ3.shape}")
print(f"X_PZ5:{X_PZ5.shape}", f"y_PZ5:{y_PZ5.shape}")

for i in range(N_wavelengths):
    mask_X = np.isnan(X_PZ3[:,i])
    X_PZ3 = np.delete(X_PZ3, mask_X, 0)
    y_PZ3 = np.delete(y_PZ3, mask_X, 0)

for i in range(N_wavelengths):
    mask_X = np.isnan(X_PZ5[:,i])
    X_PZ5 = np.delete(X_PZ5, mask_X, 0)
    y_PZ5 = np.delete(y_PZ5, mask_X, 0)
    days = np.delete(days, mask_X[:len(days)], 0)

print("\nAfter removing NaNs")
print(f"N_days:{len(days)}", f"N_wavelengths:{len(wavelengths)}")
print(f"N_points_PZ3:{len(points_PZ3)}", f"N_points_PZ5:{len(points_PZ5)}")
print(f"X_PZ3:{X_PZ3.shape}", f"y_PZ3:{y_PZ3.shape}")
print(f"X_PZ5:{X_PZ5.shape}", f"y_PZ5:{y_PZ5.shape}")
### -----------------------------------------------------------------------------------------------





### TRAINING AND TEST DATA ------------------------------------------------------------------------
X_train = np.copy(X_PZ3)
y_train = np.copy(y_PZ3)

X_validation = np.copy(X_PZ5)
y_validation = np.copy(y_PZ5)
### -----------------------------------------------------------------------------------------------





### SAVE DATA -------------------------------------------------------------------------------------
dump(np.array(days_raw), open(f"{path_input}days.sav", 'wb'))
np.save(f"{path_input}xs.npy", xs)
np.save(f"{path_input}ys.npy", ys)
np.save(f"{path_input}wavelengths.npy", wavelengths)

np.save(f"{path_input}X_train.npy", X_train)
np.save(f"{path_input}y_train.npy", y_train)
np.save(f"{path_input}X_validation.npy", X_validation)
np.save(f"{path_input}y_validation.npy", y_validation)

np.save(f"{path_input}Vr(t,x,y,wlgt).npy", db_vr_wlgt)
### -----------------------------------------------------------------------------------------------