import cv2 as cv
import matplotlib.pyplot as plt
import pydicom as dicom
from PIL import Image
from numpy import array
from sklearn.feature_extraction import image



imname = "/home/mateo/pytorch_docker/ctImages/data/raw/s_1/98.12.2/56364399.dcm"

img = dicom.dcmread(imname)
plt.imshow(img.pixel_array, cmap=plt.cm.bone)
plt.show()
img = cv.normalize(img.pixel_array, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

N_FEATURES = 1500

# SIFT
sift = cv.SIFT_create(nfeatures=N_FEATURES, nOctaveLayers=None,
                      contrastThreshold=None, edgeThreshold=None,
                      sigma=None)
kp, desc = sift.detectAndCompute(img, None)
print("SIFT Size of descriptor: ", desc.shape)
transform = cv.drawKeypoints(img,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(transform, cmap=plt.cm.bone)
plt.title("SIFT keypoints")
plt.show()
plt.imshow(desc, cmap=plt.cm.bone)
plt.title("SIFT descriptors")
plt.show()




# ORB
orb = cv.ORB_create(nfeatures=N_FEATURES, scaleFactor=None,
                    nlevels=None, edgeThreshold=None,
                    firstLevel=None, WTA_K=None,
                    scoreType=None, patchSize=None,
                    fastThreshold=None)
keypoints, desc = orb.detectAndCompute(img, None)
transform = cv.drawKeypoints(img,keypoints,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print("ORB Size of descriptor: ", desc.shape)
plt.imshow(transform, cmap=plt.cm.bone)
plt.title("ORB keypoints")
plt.show()

plt.imshow(desc, cmap=plt.cm.bone)
plt.title("ORB descriptors")
plt.show()

# Brief
dsift = cv.xfeatures2d.StarDetector_create(N_FEATURES)
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp = dsift.detect(img, None)
kp, des = brief.compute(img, kp)
print("BRIEF Size of descriptor: ", des.shape)
transform = cv.drawKeypoints(img,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(transform, cmap=plt.cm.bone)
plt.title("BRIEF keypoints")
plt.show()
plt.imshow(desc, cmap=plt.cm.bone)
plt.title("BRIEF descriptors")
plt.show()


#GIST
