import os
import nibabel as nib

pacient = '180132'
heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{pacient}/{pacient}/partes_moles_HeartSegs.nii.gz')
heart_mask = heart_segs_data.get_fdata()
print(heart_mask.shape)
print(heart_mask[heart_mask != 0].shape[0])
        
pacient_path = os.path.join('data/EXAMES/Exames_NIFTI', pacient, pacient)
nifti_files = os.listdir(pacient_path)
nifti_files.remove('partes_moles_HeartSegs.nii.gz')
# try:
motion_filename = [file for file in nifti_files if 'partes_moles_body' in file][0]
print(motion_filename)
# except:
#     print(f'partes_moles_body not found!: {pacient}')
#     break
        