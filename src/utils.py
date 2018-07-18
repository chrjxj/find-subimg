# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image

def load_img_file(fpath):
    ''' Load image from a given file path or file alike object, and convert it to a numpy ndarray object

    :param fpath: file path in file system
    :return: a numpy ndarray object
    '''

    try:
        img = Image.open(fpath)
    except IOError:
        print('load_img_file: IOError, file cannot be found; {}'.format(fpath))
        return None
    except Exception as e:
        print('load_img_file: Exception details: {}'.format(e))
        return None
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # note: convert obj to ndarray will also help make a copy in memory.
        # thus, the file-alike resource used in Image.open() could be released.
        return np.array(img)

