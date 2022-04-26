import matplotlib.pyplot as plt
import numpy as np
import math


def exponential_decay(hparam_start, hparam_end, decay_factor, step):
    value = hparam_start * math.exp(-step * decay_factor)

    return value if value >= hparam_end else hparam_end



# Agent params
learning_loops = 10
observation_ts_freq = 5

# Training params
param_iterations = 8
training_days = 14
training_epochs = 25 * param_iterations
training_epochs = 8


# Hparams
hparam_start = 0.1
hparam_end = 0.0001

decay_factor = 1e-6


x_learning_timesteps = int(learning_loops * training_epochs * training_days * 24 * 60 / observation_ts_freq)
x_timesteps = x_learning_timesteps // learning_loops

x_learning_timesteps = np.arange(x_learning_timesteps)
x_timesteps = np.arange(x_timesteps)

# exponential_decay_results = np.array(map(exponential_decay, ))
exponential_decay_results = np.array([exponential_decay(hparam_start, hparam_end, decay_factor, x) for x in x_learning_timesteps])

plt.plot(x_learning_timesteps, exponential_decay_results)
plt.show()