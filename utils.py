import numpy as np
import nibabel as nib

def get_basename(files, exclude_files, keywords):
    # Exclude files based on the exclude_files list
    files = [file for file in files if not any(f in file for f in exclude_files)]

    if gated_exam_basename := [
        file for file in files if any(keyword in file for keyword in keywords)]:
        return gated_exam_basename[0]
    else:
        raise ValueError("No matching files found.")
    
def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def set_string_parameters(avg):
    if avg > 0:
        if avg == 4:
            partes_moles_basename = 'partes_moles_FakeGated_avg_slices=4'
            avg_str = 'avg=4'
        elif avg == 3:
            partes_moles_basename = 'partes_moles_FakeGated_mean_slice=3mm'
            avg_str = 'avg=3'
    else:
        partes_moles_basename = 'partes_moles_FakeGated'
        avg_str = 'All Slices'
    return partes_moles_basename, avg_str

def calculate_area(mask, axis=2):
    mask_tmp = mask.copy()
    mask_tmp[mask_tmp != 0] = 1
    return mask_tmp.sum(axis=axis)

def create_save_nifti(data, affine, output_path):
    new_nifti = nib.Nifti1Image(data, affine)
    nib.save(new_nifti, output_path)
    