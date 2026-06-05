"""
SHD preprocessing script aligned with the original Rhythm-SNN pipeline:
- Read `shd_train.h5` / `shd_test.h5`
- Convert event streams to dense binary frames with dt=1ms
- Save each sample as `ID:{i}_{label}.npy`
"""

import os
from typing import Tuple

import numpy as np
import tables


def poisson_spikes_gen(nb_steps: int, nb_units: int, rate: float):
    spike_trains = (np.random.uniform(0, 1, (nb_steps, nb_units)) <= rate).astype(int)
    return spike_trains


def binary_image_readout(times, units, dt=1e-3):
    img = []
    N = int(1 / dt)
    times = np.array(times, copy=True)
    units = np.array(units, copy=True)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    return np.array(img)


def binary_image_readout_random(times, units, dt=1e-3, max_timestep=1000):
    img = []
    N = int(1 / dt)
    times = np.array(times, copy=True)
    units = np.array(units, copy=True)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    if N < max_timestep:
        img = np.array(img)
        pad_len = np.random.randint(0, max_timestep - N)
        head = poisson_spikes_gen(pad_len, 700, 0.01)
        tail = poisson_spikes_gen(max_timestep - N - pad_len, 700, 0.01)
        return np.vstack([head, img, tail])
    return np.array(img)


def generate_dataset(file_name: str, output_dir: str, dt=1e-3) -> int:
    fileh = tables.open_file(file_name, mode="r")
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    os.makedirs(output_dir, exist_ok=True)

    print("Number of samples: ", len(times))
    for i in range(len(times)):
        x_tmp = binary_image_readout(times[i], units[i], dt=dt)
        y_tmp = labels[i]
        output_file_name = os.path.join(output_dir, f"ID:{i}_{y_tmp}.npy")
        np.save(output_file_name, x_tmp)
    print("done..")
    fileh.close()
    return 0


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_h5 = os.path.join(root, "data", "SHD", "shd_train.h5")
    test_h5 = os.path.join(root, "data", "SHD", "shd_test.h5")
    out_train = os.path.join(root, "data", "SHD", "train_1ms")
    out_test = os.path.join(root, "data", "SHD", "test_1ms")

    if not os.path.exists(test_h5) or not os.path.exists(train_h5):
        raise FileNotFoundError("Please place shd_train.h5 and shd_test.h5 under ./data/SHD/")

    generate_dataset(test_h5, output_dir=out_test, dt=1e-3)
    generate_dataset(train_h5, output_dir=out_train, dt=1e-3)
