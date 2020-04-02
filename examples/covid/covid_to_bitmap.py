import numpy as np
from PIL import Image

from covid import COVID19

SIZE = (523, 114)


if __name__ == "__main__":
    # convert sequence to packed bit array
    conv = {"a": [False, False], "t": [False, True], "g": [True, False], "c": [True, True]}
    arr = []
    for c in COVID19:
        arr.extend(conv[c])
    byte_arr = np.array(arr, dtype="bool")

    # make an image
    img = Image.fromarray(byte_arr.astype('uint8').reshape((SIZE[1], SIZE[0])) * 255, "L")
    img.save("covid.png")
