import cv2 as cv
import matplotlib.pyplot as plt
import pydicom as dicom
import nrrd as nrrd
import numpy as np
from sklearn.feature_extraction import image

N_FEATURES = 10

imname = "/home/mateo/pytorch_docker/ctImages/data/voxels/1.2.156.14702.1.1000.16.1.2020022012121918700020003.41.5512512.nrrd"
readdata, header = nrrd.read(imname)
print(readdata.shape)
print(header)
img = cv.normalize(readdata, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
plt.imshow(img[:, : , 100], cmap=plt.cm.bone)
plt.show()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
sift = cv.xfeatures2d.SIFT_create()
orb = cv.ORB_create(nfeatures=N_FEATURES, scaleFactor=None,
                    nlevels=None, edgeThreshold=None,
                    firstLevel=0, WTA_K=2,
                    scoreType=cv.ORB_HARRIS_SCORE, patchSize=None,
                    fastThreshold=1)

total_vector = []

for i in range(0, np.shape(img)[2]):
    if i+3 <= np.shape(img)[2]:
        mask = cv.rectangle(np.zeros(np.shape(img)[:2],
                                     dtype=np.uint8),
                            (200, 150),
                            (300, 250),
                            (255),
                            thickness=-1)
        ########## ORB
        input = np.array(img[:, :, i:i+3],dtype="uint8")
        keypoints, desc = orb.detectAndCompute(input, mask)
        transform = cv.drawKeypoints(input,keypoints,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if i == 0:
            extracted_desc_orb = desc
        total_vector.extend(desc.flatten())
        #extracted_desc_orb = np.dstack([extracted_desc_orb, desc])
        plt.imshow(transform, cmap=plt.cm.bone)
        plt.title("ORB keypoints, index: {}".format(i))
        plt.show()
        ########## BRIEF
        input2 = img[:, :, i]
        kp = orb.detect(input2, mask)
        ds = brief.compute(input2, kp)[1]
        #total_vector.extend(ds.flatten())
        transform = cv.drawKeypoints(input2, kp, img)  # , flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if i == 0:
            extracted_desc_brief = ds
        #extracted_desc_brief = np.dstack([extracted_desc_brief, ds])
        plt.imshow(transform, cmap=plt.cm.bone)
        plt.title("STAR-BRIEF keypoints, index: {}".format(i))
        plt.show()

        ###### SIFT
        input3 = img[:, :, i]
        kp = orb.detect(input3, mask)
        ds = sift.compute(input3, kp)[1]
        #total_vector.extend(ds.flatten())
        transform = cv.drawKeypoints(input3, kp, img)  # , flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if i == 0:
            extracted_desc_sift = ds
        #extracted_desc_sift = np.dstack([extracted_desc_sift, ds])
        plt.imshow(transform, cmap=plt.cm.bone)
        plt.title("SIFT keypoints, index: {}".format(i))
        plt.show()
        print()

print("end")