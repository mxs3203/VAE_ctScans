import glob
import numpy as np
from skimage.transform import resize
import pickle
import cv2 as cv
import matplotlib.pyplot as plt

for f in glob.glob("{}/*.p".format("data/train/")):
    with (open(f, "rb")) as openfile:
        try:
            filename = f.split("/")[2]
            filename = filename.split(".")[:-1]
            filename = "_".join(filename)
            print(filename)
            arr = np.array(pickle.load(openfile))
            img = cv.normalize(arr[:, :, 30], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            plt.imshow(img, cmap=plt.cm.bone)
            plt.title("Before resize")
            plt.show()

            arr = resize(arr, (50, 50))
            img = cv.normalize(arr[:, :, 30], None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            plt.imshow(img, cmap=plt.cm.bone)
            plt.title("after resize")
            plt.show()
            pickle.dump(arr, open( "data/train_small/{}.p".format(filename), "wb" ) )
        except EOFError:
            print("Error")
            break