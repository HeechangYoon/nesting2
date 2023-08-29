import copy
import os
import pickle
import pathlib
import random
import math
import time

import numpy as np
import pandas as pd

class Plate():
    def __init__(self, l, b, l_min, l_max, b_min, b_max):
        self.l = l
        self.b = b
        self.pixel_l = math.ceil(self.l / 100)
        self.pixel_b = math.ceil(self.b / 100)
        self.l_min = l_min
        self.l_max = l_max
        self.b_min = b_min
        self.b_max = b_max
        self.pixel_l_max = math.ceil(self.l_max / 100)
        self.pixel_b_max = math.ceil(self.b_max / 100)
        self.Area = self.l * self.b
        self.Pixel_Area = 0
        self.PixelPlate = self.make_pixel()

    def make_pixel(self):

        plate = np.zeros((self.pixel_l, self.pixel_b))

        pixel_plate = np.pad(plate, ((0, self.pixel_l_max - self.pixel_l), (0, self.pixel_b_max - self.pixel_b)), 'constant', constant_values=1)

        self.Pixel_Area = self.pixel_l * self.pixel_b

        return pixel_plate


class Part():
    def __init__(self, Area, PixelPart):
        self.Area = Area
        self.PixelPart = PixelPart
        self.Pixel_Area = np.sum(PixelPart[:, :, 0])


def read_data():
    with open('/input/3212-A31-001_100.pickle', 'rb') as f:
        raw_data_A31 = pickle.load(f)

    with open('/input/3212-E31-001_100.pickle', 'rb') as f:
        raw_data_E31 = pickle.load(f)

    with open('/input/3212-E42-001_100.pickle', 'rb') as f:
        raw_data_E42 = pickle.load(f)

    with open('/input/3212-E52-001_100.pickle', 'rb') as f:
        raw_data_E52 = pickle.load(f)

    with open('/input/3212-F41-001_100.pickle', 'rb') as f:
        raw_data_F41 = pickle.load(f)

    raw_part_list = raw_data_A31 + raw_data_E31 + raw_data_E42 + raw_data_E52 + raw_data_F41
    raw_part_list = [Part(qq.Area, qq.PixelPart) for qq in raw_part_list]
    raw_part_list.sort(key=lambda x: -x.Area)

    for part in raw_part_list:
        part.PixelPart = np.where(part.PixelPart > 0, 1, 0)
        temp_list = list()
        for i in range(24):
            angle_part = part.PixelPart[:, :, i]

            non_zero_rows, non_zero_cols = np.nonzero(angle_part)
            start_row, start_col = non_zero_rows.min(), non_zero_cols.min()
            end_row, end_col = non_zero_rows.max() + 1, non_zero_cols.max() + 1

            temp_list.append(angle_part[start_row:end_row, start_col:end_col])
        part.PixelPart = temp_list
    return raw_part_list


def generate_data(raw_part_list,  # 샘플링 대상이 되는 픽셀 이미지 리스트
                  plate_length_min=3000,  # 생성될 강재의 최소 길이
                  plate_length_max=21000,  # 생성될 강재의 최대 길이
                  plate_breadth_min=1000,  # 생성될 강재의 최소 폭
                  plate_breadth_max=4500  # 생성될 강재의 최대 폭
                  ):

    # 강판 생성
    plate_l = np.random.uniform(plate_length_min, plate_length_max)
    plate_b = np.random.uniform(plate_breadth_min, plate_breadth_max)

    plate = Plate(plate_l, plate_b, plate_length_min, plate_length_max, plate_breadth_min, plate_breadth_max)

    # 부재 샘플링
    index_list = [i for i in range(len(raw_part_list))]
    remain_area = plate.Pixel_Area
    part_list = []
    while True:
        index_list = [i for i in index_list if (raw_part_list[i].Pixel_Area < remain_area)]
        index_list = [i for i in index_list if check_feasible(i, raw_part_list, plate)]
        if not index_list:
            break
        index = np.random.choice(index_list, replace=False)
        part = raw_part_list[index]
        remain_area -= part.Pixel_Area
        part_list.append(part)
        if remain_area < plate.Pixel_Area * 0.2:
            break

    part_list = copy.deepcopy(part_list)

    part_list.sort(key=lambda x: -x.Area)

    return plate, part_list


def check_feasible(index, raw_part_list, plate):
    temp_list = [(raw_part_list[index].PixelPart[i].shape[0] <= plate.pixel_l)
                 & (raw_part_list[index].PixelPart[i].shape[1] <= plate.pixel_b)
                 for i in range(len(raw_part_list[index].PixelPart))]
    if True in temp_list:
        return True
    else:
        return False

if __name__ == '__main__':
    raw_part_list = read_data()
    count = 0
    for _ in range(10000):
        plate, part_list = generate_data(raw_part_list)
        print(len(part_list))
