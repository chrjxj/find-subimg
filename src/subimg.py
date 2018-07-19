# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from os import path
import numpy as np
from PIL import Image
import cv2
from utils import load_img_file
# from cnn_detector import CnnDetector


def cv2_find_subimg(img_path, template_path, threshold=0.9, viz_flag=False):
    ''' Find the cropped image (template) image in another image (specified by image_path).
    If found, return bounding boxes.

    :param img_path: str, image path (image with bigger size)
    :param template_path: str, image path (image with smaller size)
    :param threshold: float, similarity threshold
    :param viz_flag: bool, save the plot for debug or not
    :return: a list of bounding box, and an image object
    '''
    img_rgb = cv2.imread(img_path)
    template = cv2.imread(template_path)
    assert (img_rgb.shape[2] == 3 and template.shape[2] == 3)
    res0 = cv2.matchTemplate(img_rgb[:, :, 0], template[:, :, 0], cv2.TM_CCOEFF_NORMED)
    res1 = cv2.matchTemplate(img_rgb[:, :, 1], template[:, :, 1], cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img_rgb[:, :, 2], template[:, :, 2], cv2.TM_CCOEFF_NORMED)

    res = (res0 + res1 + res2)/3.
    loc = np.where(res >= threshold)
    top_left_list = [pt for pt in zip(*loc[::-1])]

    h, w, _ = template.shape
    bbox_list = [(x[0], x[1], w, h) for x in top_left_list]

    if viz_flag:
        for pt in top_left_list:
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255))

    return bbox_list, img_rgb

if __name__ == '__main__':

    assert (len(sys.argv) == 3 or len(sys.argv) == 4)

    if not path.exists(sys.argv[1]) or not path.exists(sys.argv[2]):
        raise FileNotFoundError('Input files no found.')

    if not path.isfile(sys.argv[1]) or not path.isfile(sys.argv[2]):
        raise ValueError('Incorrect input value: not path to file(s).')

    img0 = load_img_file(sys.argv[1])
    img1 = load_img_file(sys.argv[2])

    # find bigger image and assume the smaller one is the template
    if img0.shape[0] > img1.shape[0] and img0.shape[1] > img1.shape[1]:
        original_img = sys.argv[1]
        template_img = sys.argv[2]
    elif img0.shape[0] < img1.shape[0] and img0.shape[1] < img1.shape[1]:
        original_img = sys.argv[2]
        template_img = sys.argv[1]
    else:
        # assume we don't accept two images with the same size
        raise ValueError('Unexpected input image size')

    print('original image path: {}'.format(original_img))
    print('template image path: {}'.format(template_img))

    if len(sys.argv) == 4:
        raise NotImplementedError('Reserve CNN version of template matching for future implementation.')
    else:
        try:
            bbox_list, _ = cv2_find_subimg(original_img, template_img, viz_flag=True)
        except:
            raise
        res = dict()
        if len(bbox_list) == 0:
            res['found'] = False
        else:
            res['found'] = True
            res['bbox'] = bbox_list

        print('result: {}'.format(res))
