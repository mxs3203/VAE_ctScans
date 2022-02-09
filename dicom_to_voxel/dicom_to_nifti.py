import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
import glob
import shutil

from dicom2nifti.exceptions import ConversionValidationError

output_dir = '/media/mateo/data1/KIRC_CT/NIFTI'
inital_path = "/media/mateo/data1/KIRC_CT/manifest-1644254639955/TCGA-KIRC/"
sampleids = glob.glob("{}*".format(inital_path))
for folder in sampleids:
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

