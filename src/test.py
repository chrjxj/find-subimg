# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
from glob import glob
import numpy as np
from PIL import Image
import cv2
import itertools
from subimg import cv2_find_sub_img
import pprint
from utils import load_img_file

def _generate_pairs(img_folder, less_flag=True):
    assert (path.exists(img_folder))
    img_paths = glob(path.join(img_folder, '*.jpg'))
    # template_paths = glob(path.join(img_folder, 'crop-*.jpg'))

    if less_flag:
        return itertools.combinations(img_paths, 2)
    else:
        return itertools.permutations(img_paths, 2)


def test(input_dir, output_dir):
    res_lines = []
    for idx, case in enumerate(_generate_pairs(input_dir)):
        res = dict()
        res['index'] = idx
        res['error'] = False

        img0_path, img1_path = case
        img0 = load_img_file(img0_path)
        img1 = load_img_file(img1_path)

        # find bigger image and assume the smaller one is the template
        if img0.shape[0] > img1.shape[0] and img0.shape[1] > img1.shape[1]:
            res['image_path'] = img0_path
            res['template_path'] = img1_path
        elif img0.shape[0] < img1.shape[0] and img0.shape[1] < img1.shape[1]:
            res['image_path'] = img1_path
            res['template_path'] = img0_path
        else:
            # assume we don't accept two images with the same size
            res['error'] = True
            res_lines.append(res)
            continue

        try:
            bbox_list, out = cv2_find_sub_img(res['image_path'], res['template_path'], threshold=0.9, viz_flag=True)
        except:
            res['error'] = True
            res_lines.append(res)
            continue

        if len(bbox_list) == 0:
            res['found'] = False
        else:
            res['found'] = True
            res['bbox'] = bbox_list
            # save the plot
            fpath = path.join(output_dir, str(idx) + '.jpg')
            cv2.imwrite(fpath, out)
        res_lines.append(res)

    with open(path.join(output_dir, 'test_results.txt'), 'w') as f:
        pprint.pprint(res_lines, f, indent=2)


if __name__ == '__main__':
    input_dir = "../images"
    output_dir = "./output"
    os.system("mkdir -p {}".format(output_dir))

    print('Start testing...')
    test(input_dir, output_dir)
    print('Finished testing, check {} for results...'.format(output_dir))
