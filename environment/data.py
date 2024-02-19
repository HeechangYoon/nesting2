import copy
import os
import pickle
import pathlib
import random
import math
import time
from scipy.ndimage import rotate

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

def generate_single_normal_distribution_integer(min_val, max_val):
    mean = (min_val + max_val) / 2
    std_dev = (max_val - min_val) / 6  # 대략적으로 전체 범위의 1/6

    while True:
        # 정규분포에 따른 랜덤 값 생성
        sample = np.random.normal(mean, std_dev)

        # 결과를 정수형으로 변환 및 범위 내 값인지 확인
        int_sample = int(round(sample))
        if min_val <= int_sample <= max_val:
            return int_sample


def generate_rec_data(plate_length=21000,
                      plate_breadth=4500
                      ):
    image_length = math.ceil(plate_length / 100)
    image_breadth = math.ceil(plate_breadth / 100)
    target_area = image_length * image_breadth * 0.9

    plate = Plate(plate_length, plate_breadth, plate_length, plate_length, plate_breadth, plate_breadth)

    # rect_length_max = math.floor(image_length * 0.5)
    # rect_breadth_max = math.floor(image_breadth * 0.5)

    # rect_length_max_list = [5, 20, 20, 20,  100]
    # rect_breadth_max_list = [5, 20, 20, 20, 20]

    shape_list = [(5,5), (5,5), (5,5), (20,20), (20,20), (20,20), (20,20), (20,20), (60,20), (80, 20), (100, 20)]

    index=0

    total_area = 0

    images = []

    while total_area < target_area:
        rect_length_max = random.choice(shape_list)[0]
        rect_breadth_max = random.choice(shape_list)[1]
        rect_length = generate_single_normal_distribution_integer(0.5*rect_length_max, rect_length_max)
        rect_breadth = generate_single_normal_distribution_integer(0.5*rect_breadth_max, rect_breadth_max)

        area = rect_length * rect_breadth

        if total_area + area > target_area:
            break

        total_area += area

        image = np.ones((rect_length, rect_breadth))

        temp_image = np.ones((rect_length, rect_breadth, 1))
        part = Part(area, temp_image)

        angle_list = list()
        for i in range(24):
            rotated_image = rotate(image, 15 * i, reshape=True, order=0)
            binary_image = np.where(rotated_image > 0.1, 1, 0)
            index += 1

            angle_list.append(binary_image)

        part.PixelPart = angle_list
        images.append(part)

    sorted_images = sorted(images, key=lambda x: x.Area, reverse=True)

    return plate, sorted_images



if __name__ == '__main__':
    raw_part_list = read_data()
    count = 0
    for _ in range(10000):
        plate, part_list = generate_data(raw_part_list)
        print(len(part_list))
