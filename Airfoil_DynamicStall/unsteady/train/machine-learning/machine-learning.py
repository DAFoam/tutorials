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
nCells = 12060
ofm = PYOFM(comm=MPI.COMM_WORLD)

# read the data
training_cases = ["pitch-05"]
float_times_1 = np.round(np.linspace(0.001, 0.8, num=800), decimals=6)
times_1 = [str(time) for time in float_times_1]
print(times_1)
float_times_2 = np.round(np.linspace(0.001, 0.7, num=700), decimals=6)
times_2 = [str(time) for time in float_times_2]
print(times_2)
features = ["PoD", "VoS", "chiSA", "PSoSS"]
inputs = None
outputs = None
for case in training_cases:
    input = []
    output = []
    if case == "pitch-04":
        times = times_1
    else:
        times = times_2
    # read inputs
    for time in times:
        case_path = os.path.join(case, time)
        print(case_path)
        input_fields = []
        for feature in features:
            field = np.zeros(nCells)
            ofm.readField(feature, "volScalarField", case_path, field)
            input_fields.append(field)
        input_fields = np.asarray(input_fields).T
        input.append(input_fields)

        # read outputs
        output_fields = []
        field = np.zeros(nCells)
        ofm.readField("betaFINuTilda", "volScalarField", case_path, field)
        output_fields.append(field)
        output_fields = np.asarray(output_fields).T
        output.append(output_fields)

    input = np.asarray(input)
    input = input.reshape(-1,input.shape[2])
    print(input.shape)
    output = np.asarray(output)
    output = output.reshape(-1,output.shape[2])
    print(output.shape)
    
    if inputs is None:
        inputs = np.copy(input)
    else:
        inputs = np.concatenate((inputs, input), axis=0)
    if outputs is None:
        outputs = np.copy(output)
    else:
        outputs = np.concatenate((outputs, output), axis=0)


print(inputs)
print(inputs.shape)
print(outputs)
print(outputs.shape)

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
        layers.Dense(units=100, activation="tanh"),
        layers.Dense(units=100, activation="tanh"),
        layers.Dense(units=1),
    ]
)

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
)

model.fit(inputs, outputs, epochs=500, batch_size=500, validation_split=0.2)

# save the model coeffs to files
model.save("dummy_nn_model")
