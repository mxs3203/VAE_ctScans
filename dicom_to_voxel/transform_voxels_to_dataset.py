import glob
import nrrd as nrrd
import cv2 as cv
import numpy as np
import pickle as p

fixedsize = 60
cutoff = 60
sizes = []
files = glob.glob("/home/mateo/pytorch_docker/ctImages/data/voxels/*.nrrd")
for f in files:
    filename = f.split("/")[6]
    filename = filename.split(".")[:-1]
    filename = "_".join(filename)
    readdata, header = nrrd.read(f)
    size = readdata.shape
    if (size[2] > cutoff ) and (size[1] == 512 and size[0] == 512):
        if size[2] == fixedsize:
            print("Saving: ", readdata.shape)
            sizes.append(readdata.shape[2])
            img = cv.normalize(readdata, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            p.dump(img, open( "train/{}.p".format(filename), "wb" ) )
        elif size[2] > fixedsize:
            print("Saving: ", readdata.shape)
            mid = int(size[2]/2)
            start = mid-(int(fixedsize/2))
            end = mid+(int(fixedsize/2))
            img = cv.normalize(readdata, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            img = img[:,:,start:end]
            print("\t", img.shape)
            p.dump(img, open("train/{}.p".format(filename), "wb"))
        else:
            print("Ignore")


print(sizes)