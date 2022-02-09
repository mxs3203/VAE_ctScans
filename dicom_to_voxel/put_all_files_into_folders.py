import os
import shutil
import glob


for i in glob.glob('/media/mateo/data1/KIRC_CT/NIFTI/*.nii.gz'):
  folders = i.split("/")
  path = "/".join(folders[:-1])
  name = folders[-1].split(".")[0]
  try:
    os.makedirs("{}/{}/".format(path, name))
    shutil.copy(i, "{}/{}".format(path, name))
  except OSError as e:
    print(i, "exists")
