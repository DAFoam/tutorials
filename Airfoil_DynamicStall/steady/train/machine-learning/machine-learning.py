import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
import tensorflow as tf
from tensorflow.keras import layers
import os

print(tf.__version__)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=6, suppress=False)

np.random.seed(0)

# Initialize pyOFM
nCells = 12236
ofm = PYOFM(comm=MPI.COMM_WORLD)


# read the data
cases = ["aoa-10", "aoa-12", "aoa-14", "aoa-16", "aoa-18"]
features = ["PoD", "VoS", "chiSA", "PSoSS"]
inputs = None
outputs = None
for case in cases:
    input = []
    output = []
    # read inputs
    for feature in features:
        field = np.zeros(nCells)
        ofm.readField(feature, "volScalarField", case, field)
        input.append(field)
    input = np.asarray(input)
    input = input.transpose()

    # read outputs
    field = np.zeros(nCells)
    ofm.readField("betaFINuTilda", "volScalarField", case, field)
    output.append(field)
    output = np.asarray(output)
    output = output.transpose()

    if inputs is None:
        inputs = np.copy(input)
    else:
        inputs = np.concatenate((inputs, input), axis=0)
    if outputs is None:
        outputs = np.copy(output)
    else:
        outputs = np.concatenate((outputs, output), axis=0)


print(inputs)
print(outputs)

# NN traning
normalizer = layers.Normalization(
    input_shape=[
        len(features),
    ],
    axis=None,
)
normalizer.adapt(inputs)

model = tf.keras.Sequential(
    [
        normalizer,
        layers.Dense(units=20, activation="tanh"),
        layers.Dense(units=20, activation="tanh"),
        layers.Dense(units=1),
    ]
)

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
)

model.fit(inputs, outputs, epochs=500, batch_size=500, validation_split=0.2)

# verify if the mse is consistent with training
#outputs_v = model.predict(inputs, verbose=0)[:, 0]
#s = (outputs - outputs_v) ** 2
#mse = np.mean(s)
#print("mse ", mse)

# save the model coeffs to files
model.save("dummy_nn_model")


#outputs_out = np.zeros_like(outputs_v, dtype="d")
#for idx, v in enumerate(outputs_v):
    #outputs_out[idx] = v

# save the predicted beta from NN for debugging
#ofm.writeField("betaFIOmegaNN_C1", "volScalarField", outputs_out[:nCells])
#ofm.writeField("betaFIOmegaNN_C2", "volScalarField", outputs_out[nCells:])
