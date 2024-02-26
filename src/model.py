"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : Fev 26, 2023
"""

from keras import Model
from keras.layers import Dense
from pickle import dump
from sklearn.metrics import r2_score, mean_squared_error





class MLP(Model):
    def __init__(self, hidden_dim=32, output_dim=1):
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation="relu")
        self.dense2 = Dense(hidden_dim, activation="relu")
        self.dense3 = Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
    def save(self, path):
        dump(self, open(path, 'wb'))

    def test(self, X, y_true):
        y_pred = self.predict(X, verbose=0)
        print(f"RMSE : {mean_squared_error(y_true=y_true, y_pred=y_pred)}")
        print(f"R2 : {r2_score(y_true=y_true, y_pred=y_pred)}")