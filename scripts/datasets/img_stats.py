import pathlib
from runstats import Statistics
import itertools
import yaml
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class ImageStats:
    def __init__(self) -> None:
        self.dims = set()
        self.count = 0
        self.sizes = Statistics()
        self.suffixes = set()
        self.lock = Lock()
    def push(self, filepath:pathlib.Path)->None:
        with Image.open(str(filepath)) as img:
            shape = np.array(img).shape
        filesize = filepath.stat().st_size

        with self.lock:
            self.dims.add(shape)
            self.count += 1
            self.sizes.push(filesize)
            self.suffixes.add(filepath.suffix)

if __name__ == '__main__':
    path = r'D:\datasets\ImageNet'

    stats = ImageStats()

    executor = ThreadPoolExecutor(max_workers=32)

    for p in itertools.chain(pathlib.Path(path).rglob('*.jp*g'),
                             pathlib.Path(path).rglob('*.png')):
        executor.submit(stats.push, p)

    print(yaml.dump(stats))

    exit(0)