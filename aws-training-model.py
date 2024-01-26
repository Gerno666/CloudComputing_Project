import tensorflow as tf
from tensorflow import keras as k
from keras.models import load_model
import joblib

# Load data
dati = joblib.load('data/dati.pkl')

# Get data
X_train = dati['X_train']
X_test = dati['X_test']
y_train = dati['y_train']
y_test = dati['y_test']

model = k.Sequential([
    k.layers.Dense(64, activation=k.activations.relu, input_shape=[30]),
    k.layers.Dense(64, activation=k.activations.relu),
    k.layers.Dense(32, activation=k.activations.relu),
    k.layers.Dense(1, activation=None)
])

print(model.summary())

model.compile(optimizer=k.optimizers.Adam(), loss=k.losses.mean_squared_error, metrics=[k.metrics.mean_absolute_error])

# Model Training
epochs=100
model.fit(X_train, y_train, epochs=epochs, batch_size=10)

model.save("AWS-tensorflow_model.h5")