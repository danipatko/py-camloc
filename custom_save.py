import numpy as np
import struct
from util import load_remap


def save_custom(name: str, map1, map2):
    with open(name + "1.sex", "wb") as f:
        shape = np.shape(map1)
        f.write(struct.pack("3i", *shape))
        for row in map1:
            for col in row:
                f.write(struct.pack(f"2h", *col))

    # NOTE: the shape's bytes must be fixed
    with open(name + "2.sex", "wb") as f:
        shape = np.shape(map2)
        f.write(struct.pack("2i", *shape))
        for row in map2:
            for col in row:
                f.write(struct.pack("H", col))


map1, map2 = load_remap("picam")
save_custom("picam", map1, map2)
