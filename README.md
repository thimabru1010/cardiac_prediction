# Chest CT CAC score estimate

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

## Preprocessing on chest CT exams
If you're evaluating the algorithm for the gated exams jump this step
### Segment with Total Segmentator
To infer the non gated exams with the neural network trained on gated exams, we need to preprocess it to make it similar to the gated exam.

First, you need to extract total segmentator bones and hear masks wich will aid us in the follwoing steps

Run the single or multiprocessing version:
```
python segment_heart.py --root_path --output_path
```
```
python segment_heart_multiprocessing.py --root_path --output_path --num_workers
```
The masks will be saved under the NIFITI folder along with the nifti exam files.
### Create the Fake Gated exam
Fake Gated is the simulation of the gated exam created from the chest CT exam.

Run the single or multiprocessing version:
```
python gen_fake_gated.py --root_path --output_path
```
```
python gen_fake_gated_multiprocessing.py --root_path --output_path --num_workers
```
A couple of files will be saved inside the NIFTI folder in this step for debug and processing purpose.
The most important file is  `non_gated_FakeGated_avg_slices=4.nii.gz` which is the fake gated exam itself.
The name means that the `non_gated_FakeGated.nii.gz`, which has the same number of slices of `non_gated.nii.gz`, was averaged with 4 slices to make the new exam has a z pixel spacing of 3.0 mm equal to the gated exam, reducing overestimation.
## Segment calcification lesions with MTAL_CACS
You can either segment the gated or the fake gated exams.

```
python cacs_predict.py --model_dir --data_dir [INPUT DIR] --prediction_dir [OUTPUT DIR] -gpu cuda --fake_gated [IF YOU'RE PROCS FAKE GATED EXAMS]
```
Three files will be created inside the nifti folder. Their filenames will be and extension of the original input filename with the following logic. Suppose the input_filename.nii.gz, the files created will be:
`[input_filename]_binary_lesion.nii.gz`, `[input_filename]_multi_lesion.nii.gz`, `[input_filename]_multi_label.nii.gz`.

## Calculate Scores

Now that we have all the necessary segmentations extracted we can calculate the Agatston Score.

In this step, we're going to estimate to agatston score from the fake gated file and compare with its reference score which is saved in a csv file.

```
python calculate_score.py --root_path [INPUT FOLDER PATH] --csv_path [CSV WITH REFS] --fake_gated [WHETHER OR NOT] --cac_th 130 [DEFAULT VALUE FOR AGATSTON SCORE]
```
A folder called `Calcium_Score_Estimations` will be created and a csv with estimated values, references and erros will be save inside it.

## Classify Scores

Now we're going to calculate some metrics and make some plots to evaluate the estimated scores.

```
python classify_scores.py --filename [CSV path with the scores] --fake_gated [WHETHER OR NOT] --avg4 [TO INFER THE AVERAGED EXAM OF FAKE GATED] --show_plots [IF U WANT TO SHOW] --output_folder [Folder name inside Experiment Metrics folder]
```
The output folder is the name of the folder inside the following path `exp_root_path = f'data/Experiments_Metrics/{exam_type}/{avg_str}/{threshold}/clssf_mode={args.clssf_mode}/{args.output_folder}`.

The results of this script will be saved inside `Experiments_Metrics`. A confusion matrix will be created along with bland altman and linear correlation plot. Metrics will be displayed inside terminal.





