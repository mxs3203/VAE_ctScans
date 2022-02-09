import glob
import nrrd as nrrd
import cv2 as cv
import numpy as np
import pickle as p

files = glob.glob("/media/mateo/data1/KIRC_CT/KIRC_NRRD/*.nrrd")
for f in files:
    filename = f.split("/")[6]
    filename = filename.split(".")[:-1]
    filename = "_".join(filename)
    readdata, header = nrrd.read(f)
    size = readdata.shape
    if (size[1] == 512 and size[0] == 512):
        depth = readdata.shape[2]
        if depth > 1:
            for d in range(depth):
                img = cv.normalize(readdata[:,:, d], None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                print("\t", img.shape)
                p.dump(img, open( "/media/mateo/data1/KIRC_CT/train_single_ch/Depth_{}_{}.p".format(d,filename), "wb" ) )


