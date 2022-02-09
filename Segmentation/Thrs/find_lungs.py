import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pydicom
import skimage.filters
from PIL import ImageDraw, Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure, morphology, draw
import pickle as p
import cv2 as cv

dcm = pydicom.read_file('/media/mateo/data1/LUAD_LUSC_CT/manifest-1644254025834/TCGA-LUAD/TCGA-17-Z011/06-30-1982-NA-NA-32248/2.000000-NA-43644/1-1.dcm')
interscept = dcm.RescaleIntercept
slope = dcm.RescaleSlope

def find_lungs(contours, min_vol=2000, max_area = 1200):
   body_and_lung_contours = []
   vol_contours = []

   for contour in contours:
      hull = ConvexHull(contour)
      #print(hull.area, hull.volume)

      # set some constraints for the volume
      if hull.volume > min_vol and hull.area < max_area:
         body_and_lung_contours.append(contour)
         vol_contours.append(hull.volume)
   #print("---")
   # Discard body contour
   if len(body_and_lung_contours) == 2:
      return body_and_lung_contours
   elif len(body_and_lung_contours) > 2:
      vol_contours, body_and_lung_contours = (list(t) for t in
                                              zip(*sorted(zip(vol_contours, body_and_lung_contours))))
      body_and_lung_contours.pop(-1)
   return body_and_lung_contours
def create_mask_from_polygon(image, contours):
   lung_mask = np.array(Image.new(mode='L', size=image.shape))
   for contour in contours:
      x = contour[:, 0]
      y = contour[:, 1]
      polygon_tuple = list(zip(x, y))
      img = Image.new('L', image.shape, 0)
      ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
      mask = np.array(img)
      lung_mask += mask

   lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
   return lung_mask.T

image = p.load(open("/media/mateo/data1/LUAD_LUSC_CT/train/1_3_6_1_4_1_14519_5_2_1_7777_9002_212063566717885658141795941262_2551251219820905.p",'rb'))
depth = image.shape[2]
for d in range(depth):

   img = image [:,:440 ,d]/255.0
   img = cv.blur(img, (40,40))
   t = skimage.filters.threshold_otsu(img)
   cnts = measure.find_contours(img, t)

   lungs = find_lungs(cnts, min_vol=2000, max_area=1200)
   mask = create_mask_from_polygon(img, lungs)
   f, axarr = plt.subplots(1,3)
   axarr[0].imshow(img, 'gray')
   axarr[1].imshow(img, 'gray')
   axarr[2].imshow(mask)
   plt.show()