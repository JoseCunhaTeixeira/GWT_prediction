"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Fev 26, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

from folders import path_models, path_output, path_input





days = load(open(f"{path_input}days.sav", 'rb'))
N_samples = len(days); date_start = days[0]; date_end = days[-1]

xs = np.load(f"{path_input}xs.npy")
ys = np.load(f"{path_input}ys.npy")

wavelengths = np.load(f"{path_input}wavelengths.npy")
N_wavelenghts = len(wavelengths)

y_PZ3 = np.load(f"{path_input}GWT_PZ3(t).npy")
y_PZ5 = np.load(f"{path_input}GWT_PZ5(t).npy")

model_name = "20240124-1903_MLP_trainPZ3"
model = load(open(f"{path_models}{model_name}.sav", 'rb'))

db_vr_wlgt = np.load(f"{path_input}Vr(t,x,y,wlgt).npy")

points_PZ5 = [(93.00,19.00), (96.00,19.00), (99.00,19.00),
              (93.00,14.25), (96.00,14.25), (99.00,14.25),
              ]
points_PZ3 = [(63.00,04.75), (66.00,04.75), (69.00,04.75),
              (63.00,00.00), (66.00,00.00), (69.00,00.00),
             ]





### COMPUTE predictions ---------------------------------------------------------------------------
db_piezo = np.ndarray((len(days), len(xs), len(ys)), dtype=np.double)
db_piezo[:,:,:] = np.nan

print("\nStarting prediction calculation")

for x_i, x in zip( tqdm(range(len(xs)), colour='blue', initial=1, leave=False), xs ):
    for y_i, y in zip( tqdm(range(len(ys)), colour='blue', initial=1, leave=False), ys ):
        mask_isnan = np.isnan(db_vr_wlgt[:, x_i, y_i, 0])
        for day_i, day in enumerate(days) :
            if mask_isnan[day_i] == False :
                X = np.copy(db_vr_wlgt[day_i, x_i, y_i, :])
                X = np.reshape(X, (1,X.shape[0]))
                prediction = model.predict(X, verbose=0)
                db_piezo[day_i, x_i, y_i] = prediction

print("Prediction calculation ended")
### -----------------------------------------------------------------------------------------------





### SAVE predictions ------------------------------------------------------------------------------
filename = f"{path_output}{model_name}_GWT(t,x,y).npy"
np.save(filename, db_piezo)
### -----------------------------------------------------------------------------------------------





# ---





### LOAD predictions ------------------------------------------------------------------------------
db_piezo = np.load(f"{path_output}{model_name}_GWT(t,x,y).npy")
db_piezo = -db_piezo
### -----------------------------------------------------------------------------------------------





### TEST PREDICTIONS ------------------------------------------------------------------------------
for p_i, point in enumerate(points_PZ3):
    print(f"\nPoint {point}")
    X_PZ3 = db_vr_wlgt[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0], :]
    mask_X = np.isnan(X_PZ3[:,0])
    X_PZ3 = np.delete(X_PZ3, mask_X, 0)

    y_pred = db_piezo[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0]]
    print(f"RMSE : {mean_squared_error(y_true=-y_PZ3[~mask_X], y_pred=-y_pred[~mask_X])}")
    print(f"R2 : {r2_score(y_true=-y_PZ3[~mask_X], y_pred=-y_pred[~mask_X])}")

    fig, ax = plt.subplots(figsize=(16,5), dpi=300)
    ax.plot(days, y_PZ3, c='green')
    ax.plot(days, y_pred, c='orange')
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("GWT level (m)")
    plt.tight_layout()
    ax.legend(["Real", "Predicted"])
    ax.set_ylim([-4,-1])
    ax.set_title(f"point {point} | R² {r2_score(y_true=-y_PZ3[~mask_X], y_pred=-y_pred[~mask_X])} | RMSE {mean_squared_error(y_true=-y_PZ3[~mask_X], y_pred=-y_pred[~mask_X])}")
    fig.savefig(f"{path_output}point{point}_GWTprediction_fromMap.png", format='png', dpi='figure', bbox_inches='tight')

print()

for p_i, point in enumerate(points_PZ5):
    print(f"\nPoint {point}")
    X_PZ5 = db_vr_wlgt[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0], :]
    mask_X = np.isnan(X_PZ5[:,0])
    X_PZ5 = np.delete(X_PZ5, mask_X, 0)

    y_pred = db_piezo[:, np.where(xs==point[0])[0][0], np.where(ys==point[1])[0][0]]
    print(f"RMSE : {mean_squared_error(y_true=-y_PZ5[~mask_X], y_pred=-y_pred[~mask_X])}")
    print(f"R2 : {r2_score(y_true=-y_PZ5[~mask_X], y_pred=-y_pred[~mask_X])}")

    fig, ax = plt.subplots(figsize=(16,5), dpi=300)
    ax.plot(days, y_PZ5, c='green')
    ax.plot(days, y_pred, c='orange')
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("GWT level (m)")
    plt.tight_layout()
    ax.legend(["Real", "Predicted"])
    ax.set_ylim([-4,-1])
    ax.set_title(f"point {point} | R² {r2_score(y_true=-y_PZ5[~mask_X], y_pred=-y_pred[~mask_X])} | RMSE {mean_squared_error(y_true=-y_PZ5[~mask_X], y_pred=-y_pred[~mask_X])}")
    fig.savefig(f"{path_output}point{point}_GWTprediction_fromMap.png", format='png', dpi='figure', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------