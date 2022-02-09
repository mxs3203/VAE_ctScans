import os

from radiomics import featureextractor, getTestCase


params = os.path.join("pyrad_params.yaml")
extractor = featureextractor.RadiomicsFeatureExtractor(params)
print(extractor.enabledFeatures)

result = extractor.execute("/media/mateo/data1/kits19/data/case_00000/imaging.nii.gz",
                           "/media/mateo/data1/kits19/data/case_00000/segmentation.nii.gz",
                           voxelBased=False)

print(result)

