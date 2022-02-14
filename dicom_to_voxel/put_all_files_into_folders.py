import os
import shutil
import glob

from tqdm import tqdm

for i in tqdm(glob.glob('/media/mateo/data1/LUAD_LUSC_CT/NIFTI/*.nii.gz')):
  folders = i.split("/")
  path = "/".join(folders[:-1])
  name = folders[-1].split(".")[0]
  try:
    os.makedirs("{}/{}/".format(path, name))
    shutil.move(i, "{}/{}".format(path, name))
  except OSError as e:
    print(i, "exists")
