from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image

import numpy as np
import os
import glob
import math
import tqdm
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm

PIC_FOLDER = 'custom_data/compositions/'
BOX = {'linewidth':1, 'edgecolor':'g', 'facecolor':'none'}

def build_win():
	pic = Image.open(PIC_FOLDER + 'man.jpg')
	im = np.array(pic, dtype=np.uint8)
	fig,ax = plt.subplots(1)
	ax.imshow(im)
	rect = patches.Rectangle((500,200), 300, 250, **BOX)
	ax.add_patch(rect)
	plt.text(550, 180, 'Mischa')
	plt.show()

def __main():
	build_win()

if __name__ == '__main__':
	__main()