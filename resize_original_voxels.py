import glob
import numpy as np
from skimage.transform import resize
import pickle

for f in glob.glob("{}/*.p".format("data/train/")):
    with (open(f, "rb")) as openfile:
        try:
            filename = f.split("/")[2]
            filename = filename.split(".")[:-1]
            filename = "_".join(filename)
            arr = np.array(pickle.load(openfile))
            arr = resize(arr, (50, 50))
            pickle.dump(arr, open( "data/train_small/{}.p".format(filename), "wb" ) )
        except EOFError:
            break