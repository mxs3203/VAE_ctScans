import os

import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
import glob
import shutil
import argparse

from dicom2nifti.exceptions import ConversionValidationError


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--Output", help = "Output")
parser.add_argument("-i", "--Input", help = "Input")
parser.add_argument("-c", "--CheckExisting", help = "Check Existing Files")
args = parser.parse_args()
print(args)

output_dir = args.Output #'/media/mateo/data1/TCGA-LUAD_LUSC_CT/NIFTI/'
inital_path = args.Input #"/media/mateo/data1/LUAD_LUSC_CT/manifest-1644254025834/TCGA-LUSC/"
CHECK_EXISTING = args.CheckExisting #True

if CHECK_EXISTING:
    # Let's See if there is already some NII.GZ files so we can skip them
    done = glob.glob("{}*.nii.gz".format(output_dir))
    done_ids = [d.split("/")[6].split("_")[0] for d in done]

    # Construct new list of files to process based on what is already done
    todo_sampleids = []
    sampleids = glob.glob("{}*".format(inital_path))
    for id in sampleids:
        id_Str = id.split("/")[-1]
        if id_Str in done_ids:
            todo_sampleids.append(id)
else:
    todo_sampleids = glob.glob("{}*".format(inital_path))

for folder in todo_sampleids:
    sampleid = folder.split("/")[-1].strip()
    print(sampleid)
    person_scans = glob.glob("{}/*".format(folder))
    for s in person_scans:
        date = s.split("/")[-1].split(" ")[0].replace("-","")
        indv_scans = glob.glob("{}/*".format(s))
        if len(indv_scans):
            for ss in indv_scans:
                if len(glob.glob("{}/*.dcm".format(ss))) > 2:
                    scanname2 = ss.split("/")[-1].replace(" ","").replace("-", "_").replace(".","_")
                    nifti_name = "{}/{}_{}_{}".format(output_dir,sampleid, date, scanname2).strip()
                    print("\t",scanname2)
                    try:
                        dicom2nifti.convert_directory(ss,ss)
                        new_file = glob.glob("{}/*.nii.gz".format(ss))
                        if len(new_file) > 0:
                            shutil.move(new_file[0], "{}.nii.gz".format(nifti_name))
                    except ConversionValidationError:
                        print("Inconsistency in slides, skipping...", scanname2)



for i in glob.glob(args.Output):
  folders = i.split("/")
  path = "/".join(folders[:-1])
  name = folders[-1].split(".")[0]
  try:
    os.makedirs("{}/{}/".format(path, name))
    shutil.copy(i, "{}/{}".format(path, name))
  except OSError as e:
    print(i, "exists")
