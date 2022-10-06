import os
import re
import shutil
import cv2
import numpy as np
from collections import Counter
from numba import jit


@jit(nopython=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass


def calcEntropy2d(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # traversal computing two-tuples
    IJ = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)

    Fij = Counter(IJ).items()

    # Calculate the probability of occurrence of each two-tuples
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H


def consOv(filepath, image_list, ov):

    if not os.path.exists(ov):
        os.makedirs(ov)

    for file_path, empty_list, file_name_list in os.walk(filepath):

        for file_name in file_name_list:
            for image_name in image_list:
                # regular match
                if re.match(image_name, file_name):
                    oldfile = file_path + file_name
                    shutil.copy(oldfile, ov)


def main():

    # M40 M10
    H_rank = {}

    # Calculate information entropy
    for i in range(1, 21):

        # M40: car_0001 M10: bed_00001
        vi = "car_0001_" + "{:0>3d}".format(i) + ".png"
        img = cv2.imread("../data/calcEntropy/M40/" + vi, cv2.IMREAD_GRAYSCALE)
        Hi = calcEntropy2d(img, 3, 3)
        Hi = "{:.4f}".format(float(Hi))

        H_rank.update({i: Hi})

    # Information entropy ranking
    H_rank = dict(sorted(H_rank.items(), key=lambda item: item[1], reverse=True))

    file = open('../data/calcEntropy/M40/entropyRank.txt', 'w')
    n = 0
    v1_list = []
    vi_list = []

    # Information entropy rankings are saved in dictionaries and text files
    for k, v in H_rank.items():
        n = n + 1
        file.write("viewpoint number: " + str(k) + '    ' + "information entropy: " + str(v) + '    ' + "rank: " + str(
            n) + '\n')

        # M40: car_0001 M10: bed_00001
        v1 = "car_0001_" + "{:0>3d}".format(k) + ".png"
        vi = '\w+\_\d+\_' + "{:0>3d}".format(k) + ".png"

        # Set the number of views n(n = 1, 2, ..., 20)
        if n <= 6:
            v1_list.append(v1)
            vi_list.append(vi)

    print(v1_list)
    print(vi_list)
    file.close()

    # Construct the optimal viewset
    print("========Optimal view set generation has started========")

    # M40
    class_list = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
                  'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
                  'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
                  'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
                  'wardrobe', 'xbox']

    # M10
    # class_list = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    mode_list = ['train', 'test']

    for class_name in class_list:
        for mode_name in mode_list:

            # Original 20-view dataset
            filepath = '../data/m40_20/' + class_name + '/' + mode_name + '/'
            # optimal viewset
            ov = '../data/m40_v6(r1-6)/' + class_name + '/' + mode_name + '/'

            consOv(filepath, vi_list, ov)

    print("========The optimal view set has been generated========")


if __name__ == '__main__':

    main()
