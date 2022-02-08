import glob
import pydicom as dicom
import pickle as p
import numpy as np
import pandas as pd
import pydicom.uid
import matplotlib.pyplot as plt

CT_SCAN_SIZE = 512
inital_path = "/media/mateo/data1/LUAD_LUSC_CT/manifest-1644254025834/TCGA-LUAD/"
all_folders = glob.glob("{}*".format(inital_path))

data_manifest = pd.DataFrame([])

for patientID in all_folders: # for every patient folder
    print('ID: ',patientID)
    days = glob.glob("{}/*".format(patientID))
    for d in days: # for every day of that ID
        #print('Day: ',d)
        scans = glob.glob("{}/*".format(d))
        for s in scans:
            #print('Scan: ', s)
            images = glob.glob("{}/*.dcm".format(s))
            image = np.zeros(shape=(CT_SCAN_SIZE, CT_SCAN_SIZE))
            valid = False
            ds_obj = None
            for im in sorted(images):
                #print('Image: ', im)
                try:
                    ds = dicom.read_file(im, force=False)
                    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                    img = ds.pixel_array  # dtype = uint16
                    ds_obj = ds
                    image = np.dstack([image, img])
                    #print("\t",ds.AccessionNumber)
                    valid = True
                except (pydicom.errors.InvalidDicomError, ValueError):
                    valid = False
                    print("Oops!  That was no DCIM... ")
            if valid:
                image = image[:,:,-1] # remove the first zeros
                with open(r"{}/VOXEL_{}_{}_{}.p".format(s, ds_obj.PatientID, ds_obj.AccessionNumber,ds_obj.AcquisitionDate), "wb") as output_file:
                    p.dump(image,output_file)
                tmp_df = pd.DataFrame({"PatientID": ds_obj.PatientID, "AccessionNumber": ds_obj.AccessionNumber,"AcquisitionDate":ds_obj.AcquisitionDate,
                                      "AcquisitionNumber":ds_obj.AcquisitionNumber, "BodyPartExamined":ds_obj.BodyPartExamined,
                                      "ContentDate": ds_obj.ContentDate, "StudyInstanceUID":ds_obj.StudyInstanceUID,
                                      "path":"{}/VOXEL_{}_{}_{}.p".format(s,ds_obj.PatientID, ds_obj.AccessionNumber,ds_obj.AcquisitionDate)},  index=[0])
                data_manifest = data_manifest.append(tmp_df, ignore_index=True )


print(data_manifest)
data_manifest.to_csv('manif.csv')