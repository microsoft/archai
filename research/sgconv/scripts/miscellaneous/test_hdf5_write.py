import h5py
import numpy as np
from tqdm import tqdm


def main():
    f = h5py.File("test.hdf5", "w")

    num_rows = 500000
    for i in range(num_rows):
        if i % 10000 == 0:
            print(f"creating {i}")
        this_data = np.float32(np.random.rand(1, 51200))
        f.create_dataset(str(i), data=this_data)

    f.close()


if __name__ == "__main__":
    main()
