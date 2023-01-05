import h5py
import numpy as np
from tqdm import tqdm


def main():
    f = h5py.File("test.hdf5", "r")

    data_storage = []
    for ind, key in enumerate(f.keys()):
        if ind % 10000 == 0:
            print(f"handling {key}")
        this_data = np.float32(np.array(f[key]))
        data_storage.append(this_data)

    f.close()

    features = np.concatenate(data_storage, axis=0)
    np.save("test.npy", features)
    print("done")


if __name__ == "__main__":
    main()
