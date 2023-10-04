import numpy as np
import random
import itertools
import pandas as pd
import math
from environment.data import read_data, generate_data

from collections import OrderedDict


class Management():
    def __init__(self, plate, part_list):
        self.plate = plate
        self.part_list = part_list
        self.part_num = len(self.part_list)
        self.batch_num = 0
        self.step = 0

    def assign(self, action):
        overlap = False
        raw_part = self.part_list.pop(0)
        part = raw_part.PixelPart[action[1]]

        pixel_x = action[0]

        plate_rows, plate_cols = self.plate.PixelPlate.shape
        part_rows, part_cols = part.shape

        temp = np.zeros(self.plate.PixelPlate.shape)
        temp += self.plate.PixelPlate

        ref_point = (0, 0)
        for pixel_y in range(plate_cols - part_cols + 1):
            overlap = False
            temp = np.zeros(self.plate.PixelPlate.shape)
            temp += self.plate.PixelPlate

            if (pixel_x + part_rows <= plate_rows) & (pixel_y + part_cols <= plate_cols):
                temp[pixel_x:pixel_x + part_rows, pixel_y:pixel_y + part_cols] += part
                ref_point = (pixel_x + part_rows, pixel_y + part_cols)
            else:
                overlap = True
            if not overlap:
                if np.max(temp) > 1:
                    overlap = True
                else:
                    overlap = False
                    self.plate.PixelPlate = temp
                    self.batch_num += 1
            if not overlap:
                break

        self.step += 1
        return overlap, temp, ref_point


if __name__ == '__main__':
    raw_part_list = read_data()
    plate, part_list = generate_data(raw_part_list)

    nest = Management(plate, part_list)
    nest.assign((100, 0, 10))
