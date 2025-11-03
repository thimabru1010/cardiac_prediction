# chest CT CAC score estimate

This repository aim to estimate the CAC score of a chest CT non-gated exam.

To do this, we use a segmentation neural network, trained on gated exams, to find the lesions. Knowing that the chest CT exam is very different from the gated CT exam we preprocessed the chest CT exam to simulate a gated CT exam and use it in the neural network.

Down Below I show the folder structure and set the correct order to run the scripts to estimate the chest CT CAC score and compare it with the gated exam as well.

## Folder Structure

First of all, we need to convert the DICOM files to NIFTI.

You can convert DICOM to NIFTI using the `dcm2nifti.py` script.

```
DICOM_Data
  |-- patient_id01
      |-- G1.dcm
      |-- G2.dcm
      ...
      |-- SG1.dcm
      |-- SG2.dcm
      ...
  |-- patient_id02
  ...
```
Where Gi is the slice i of gated exam and SGi is the slice i of non-gated exam.

A new output folder will be created, defined inside the script, following the same structure but now with the nifti files.
```
DICOM_Data
  |-- patient_id01
      |-- [patient_id]_gated.nii.gz
      |-- [patient_id]_non_gated.nii.gz
  |-- patient_id02
      |-- [patient_id]_gated.nii.gz
      |-- [patient_id]_non_gated.nii.gz
  ...
```
