"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Fev 26, 2023
"""

import matplotlib.pyplot as plt
from datetime import datetime
from numpy import load
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from folders import path_models, path_input
from model import MLP





X_train = load(f"{path_input}X_train.npy")
y_train = load(f"{path_input}y_train.npy")
X_validation = load(f"{path_input}X_validation.npy")
y_validation = load(f"{path_input}y_validation.npy")

model = MLP()
model.compile(optimizer = Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mean_absolute_error'])

history = model.fit(X_train, y_train, epochs=500, batch_size=2, validation_data=(X_validation, y_validation), shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=100)])

model.summary()

model.save(f"{path_models}{datetime.now().strftime('%Y%m%d-%H%M')}_MLP.sav")

fig, ax = plt.subplots(figsize=(16,5), dpi=300)
epochs = range(len(history.history['loss']))
ax.semilogy(epochs, history.history['loss'], epochs, history.history['val_loss'])
ax.set_xlabel("Epochs")
ax.set_ylabel("RMSE")
ax.legend(["Training dataset", "Validation dataset"])
fig.savefig(f"{path_models}{datetime.now().strftime('%Y%m%d-%H%M')}_history.png", format='png', dpi='figure', bbox_inches='tight')

print("\nEvaluation on training dataset")
y_pred = model.test(X_train, y_train)

print("\nEvaluation on test dataset")
y_pred = model.test(X_validation, y_validation)