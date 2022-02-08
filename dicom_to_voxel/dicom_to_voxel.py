#!/usr/bin/env python

import sys
import os
import itk
import glob
import argparse

parser = argparse.ArgumentParser(description="Read DICOM Series And Write 3D Image.")
parser.add_argument(
    "dicom_directory",
    nargs="?",
    help="If DicomDirectory is not specified, current directory is used",
)
parser.add_argument("output_image", nargs="?")
parser.add_argument("series_name", nargs="?")
args = parser.parse_args()

inital_path = "/media/mateo/data1/KIRC_CT/manifest-1644254639955/TCGA-KIRC/"
all_folders = glob.glob("{}*".format(inital_path))
for folder in all_folders:
    ff = glob.glob("{}/*".format(folder))
    for f in ff:
        f = glob.glob("{}/*".format(f))
        if len(f):
            print(f[0])
            # current directory by default
            dirName = "{}/".format(f[0])
            if args.dicom_directory:
                dirName = args.dicom_directory

            PixelType = itk.ctype("signed short")
            Dimension = 3

            ImageType = itk.Image[PixelType, Dimension]

            namesGenerator = itk.GDCMSeriesFileNames.New()
            namesGenerator.SetUseSeriesDetails(True)
            namesGenerator.AddSeriesRestriction("0008|0021")
            namesGenerator.SetGlobalWarningDisplay(False)
            namesGenerator.SetDirectory(dirName)

            seriesUID = namesGenerator.GetSeriesUIDs()

            if len(seriesUID) < 1:
                print("No DICOMs in: " + dirName)


            print("The directory: " + dirName)
            print("Contains the following DICOM Series: ")
            for uid in seriesUID:
                print(uid)

            seriesFound = False
            for uid in seriesUID:
                seriesIdentifier = uid
                if args.series_name:
                    seriesIdentifier = args.series_name
                    seriesFound = True
                print("Reading: " + seriesIdentifier)
                fileNames = namesGenerator.GetFileNames(seriesIdentifier)

                reader = itk.ImageSeriesReader[ImageType].New()
                dicomIO = itk.GDCMImageIO.New()
                reader.SetImageIO(dicomIO)
                reader.SetFileNames(fileNames)
                reader.ForceOrthogonalDirectionOff()

                writer = itk.ImageFileWriter[ImageType].New()
                outFileName = os.path.join(dirName, folder,seriesIdentifier + ".nrrd")
                if args.output_image:
                    outFileName = args.output_image
                writer.SetFileName(outFileName)
                writer.UseCompressionOn()
                writer.SetInput(reader.GetOutput())
                print("Writing: " + outFileName)
                writer.Update()

                if seriesFound:
                    print("Nothing foud... moving on")

# find TCGA-LUSC/ -name "*.nrrd" -exec mv "{}" LUSC-NRRD/ \;