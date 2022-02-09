import pickle as p
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import skimage
import skimage.filters

img = p.load(open("/media/mateo/data1/KIRC_CT/train_single_ch/Depth_10_1_3_6_1_4_1_14519_5_2_1_1706_4004_636512598335971651218546017347_35_00000051251219950201.p",'rb'))

# plt.imshow(img, cmap='gray'   )
# plt.show()
#
blurred_image = skimage.filters.gaussian(img, sigma=5)

fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap='gray')
plt.show()

histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

t = skimage.filters.threshold_otsu(blurred_image)
#t = 0.65
binary_mask = blurred_image < t

fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap='gray')
plt.show()

selection = np.zeros_like(img)
selection[binary_mask] = img[binary_mask]

fig, ax = plt.subplots()
plt.imshow(selection,  cmap='gray')
plt.show()

